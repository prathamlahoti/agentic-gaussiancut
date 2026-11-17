"""Frozen post-edit probes. This job has no route back into the agent's decisions."""
import os,sys,json,subprocess,traceback,gc,shutil,inspect
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT=Path('/kaggle/working/evaluation');ROOT.mkdir(parents=True,exist_ok=True)
BASE=next(Path('/kaggle/input').rglob('v2/report.json')).parent.parent
CFG=json.loads((BASE/'v2_config.json').read_text())
EVALUATION_PROMPTS={'room_footstool':'footstool. ottoman.','room_table':'table.','counter_plant':'potted plant.'}


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2))


def command(args):subprocess.run([str(x) for x in args],check=True)


def main():
    command([sys.executable,'-m','pip','install','--quiet','transformers==4.51.3','timm','git+https://github.com/facebookresearch/segment-anything.git@'+CFG['sam_revision'],'git+https://github.com/facebookresearch/sam2.git@'+CFG['sam2_revision']])
    from segment_anything import sam_model_registry,SamPredictor,SamAutomaticMaskGenerator
    from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from improved import backproject,projection,sam_mask,mask_quality,overlay
    from benchmark_metrics import depth_pair,score_arm
    # The helper has the same detector settings as the editing job, only for
    # pre-edit pseudo-reference creation. Evaluation below has its own frozen probe.
    import improved
    def detect(image,prompt,processor,model):
        inputs=processor(images=image,text=prompt,return_tensors='pt').to('cuda')
        with torch.inference_mode():out=model(**inputs)
        key='threshold' if 'threshold' in inspect.signature(processor.post_process_grounded_object_detection).parameters else 'box_threshold'
        r=processor.post_process_grounded_object_detection(out,inputs.input_ids,**{key:.30},text_threshold=.25,target_sizes=[(image.height,image.width)])[0]
        return {'boxes':r['boxes'].cpu().tolist(),'scores':r['scores'].cpu().tolist()}
    improved.detect=detect
    import urllib.request
    w=Path('/kaggle/working/sam_vit_h.pth')
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',w)
    sam=sam_model_registry['vit_h'](checkpoint=str(w)).to('cuda').eval()
    predictor=SamPredictor(sam);auto=SamAutomaticMaskGenerator(sam)
    processor=AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny')
    detector=AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny').to('cuda').eval()
    arms=[]
    eligible_views={}
    for case in CFG['cases']:
        src=BASE/'v2'/case['id'];sc=BASE/'v2/scenes'/case['scene']
        if not (src/'seed_mask.png').exists():continue
        records=json.loads((sc/'cameras.json').read_text());seed=case['seed']
        sm=np.asarray(Image.open(src/'seed_mask.png'))>0;sd=np.load(sc/(seed+'.npy'))
        points=backproject(records[seed],sm,sd)
        refs=ROOT/case['id']/'references';refs.mkdir(parents=True,exist_ok=True)
        eligible_views[case['id']]=[]
        exclusions=[];partial_views=[];complete_views=[]
        for ident in case['held_out']:
            image=Image.open(sc/(ident+'.png')).convert('RGB');depth=np.load(sc/(ident+'.npy'))
            proj=projection(points,records[ident],depth)
            if proj is None:
                exclusions.append({'view':ident,'reason':'No visible seed-point projection in original render'})
                continue
            proposals,det=sam_mask(image,proj,predictor,detector,processor,case['prompt'])
            # Rank reference proposal BEFORE any post-edit images are inspected.
            rated=[]
            for m,score,box in proposals:
                q=mask_quality(m,proj,records[ident],depth,records[seed],sm,sd)
                rated.append((q['seed_coverage']+q['reverse_agreement']+score,m,q))
            _,mask,q=max(rated,key=lambda x:x[0])
            Image.fromarray(mask.astype('uint8')*255).save(refs/(ident+'.png'))
            overlay(image,mask).save(refs/(ident+'_overlay.png'))
            save(refs/(ident+'.json'),{'origin':'SAM-H pseudo reference from original render only','needs_visual_review':True,'quality':q})
            if q['seed_coverage']<.60 or mask.mean()<.001:
                exclusions.append({'view':ident,'reason':'Unsupported pre-edit pseudo-reference','quality':q})
                continue
            eligible_views[case['id']].append(ident)
            (partial_views if q['clipped'] else complete_views).append(ident)
            segments=auto.generate(np.asarray(image))
            stack=np.stack([m['segmentation'] for m in segments]) if segments else np.empty((0,image.height,image.width),bool)
            np.savez_compressed(refs/(ident+'_sam.npz'),masks=stack)
        save(refs/'eligibility.json',{'eligible':eligible_views[case['id']],'complete_views':complete_views,'partial_views':partial_views,'excluded':exclusions,'rule':'Pre-edit only: exclude coverage below .60 or area below .001. Partial targets are retained in the declared pilot, flagged, and excluded from a separate complete-view summary. Paper G.3 excludes incomplete target coverage.'})
        variants=[(policy,BASE/'removal_pipeline/runs'/(case['id']+'_'+policy)) for policy in ['fixed','vlm']]
        if case['id']=='room_footstool':
            old=list(Path('/kaggle/input').rglob('runs/agentic_room/manifest.json'))
            if old:variants.append(('original_v1',old[0].parent))
        for policy,run in variants:
            if not (run/'manifest.json').exists():continue
            dst=ROOT/case['id']/policy
            for sub in ['masks','rgb/before','rgb/after','sam/before','sam/after','gsam2/before/mask','gsam2/after/mask']:(dst/sub).mkdir(parents=True,exist_ok=True)
            depth_meta={}
            if not eligible_views[case['id']]:
                print('NO_ELIGIBLE_EVAL_VIEWS',case['id'],flush=True)
                continue
            for ident in eligible_views[case['id']]:
                shutil.copy2(refs/(ident+'.png'),dst/'masks'/(ident+'.png'))
                shutil.copy2(refs/(ident+'_sam.npz'),dst/'sam/before'/(ident+'.npz'))
                for phase in ['before','after']:shutil.copy2(run/'rgb'/phase/(ident+'.png'),dst/'rgb'/phase/(ident+'.png'))
                im=Image.open(dst/'rgb/after'/(ident+'.png')).convert('RGB')
                segments=auto.generate(np.asarray(im))
                stack=np.stack([m['segmentation'] for m in segments]) if segments else np.empty((0,im.height,im.width),bool)
                np.savez_compressed(dst/'sam/after'/(ident+'.npz'),masks=stack)
                a=np.load(run/'depth/before'/(ident+'.npy'));b=np.load(run/'depth/after'/(ident+'.npy'))
                _,meta=depth_pair(a,b,dst,ident)
                target=np.asarray(Image.open(refs/(ident+'.png')))>0
                meta['valid_target_fraction']=float((np.isfinite(a)&np.isfinite(b)&(a>0)&(b>0))[target].mean())
                depth_meta[ident]=meta
            save(dst/'depth_adapter.json',depth_meta)
            arms.append((case,dst))
            print('STRUCTURE_DONE',case['id'],policy,flush=True)
    del sam,predictor,auto;gc.collect();torch.cuda.empty_cache()
    predictor2=SAM2ImagePredictor.from_pretrained('facebook/sam2.1-hiera-large',device='cuda')
    # Mirrors released Grounded-SAM-2 HF demo: tiny detector, .4/.3 thresholds,
    # SAM2.1 large, box prompt, multimask_output=False. Float32 on T4 (no BF16).
    results=[]
    for case,dst in arms:
        predictions=[]
        for ident in eligible_views[case['id']]:
            for phase in ['before','after']:
                im=Image.open(dst/'rgb'/phase/(ident+'.png')).convert('RGB')
                inputs=processor(images=im,text=EVALUATION_PROMPTS[case['id']],return_tensors='pt').to('cuda')
                with torch.inference_mode():out=detector(**inputs)
                key='threshold' if 'threshold' in inspect.signature(processor.post_process_grounded_object_detection).parameters else 'box_threshold'
                r=processor.post_process_grounded_object_detection(out,inputs.input_ids,**{key:.4},text_threshold=.3,target_sizes=[(im.height,im.width)])[0]
                boxes=r['boxes'].cpu().numpy();mask=np.zeros((im.height,im.width),bool)
                if len(boxes):
                    with torch.inference_mode():
                        predictor2.set_image(np.asarray(im));ms,_,_=predictor2.predict(box=boxes,multimask_output=False)
                    if ms.ndim==4:ms=ms[:,0]
                    if ms.ndim==2:ms=ms[None]
                    mask=ms.astype(bool).any(0)
                Image.fromarray(mask.astype('uint8')*255).save(dst/'gsam2'/phase/'mask'/(ident+'.png'))
                predictions.append({'view':ident,'phase':phase,'boxes':boxes.tolist(),'scores':r['scores'].cpu().tolist()})
        save(dst/'semantic_probe.json',{'model':'GroundingDINO tiny + SAM2.1 large','category_prompt':EVALUATION_PROMPTS[case['id']],'box_threshold':.4,'text_threshold':.3,'full_frame':True,'predictions':predictions})
        scores=score_arm(dst);results.append({'case':case['id'],'policy':dst.name,'metrics':scores})
        save(ROOT/'summary.json',results);print('EVALUATED',case['id'],dst.name,json.dumps(scores),flush=True)
    w.unlink()


if __name__=='__main__':
    try:main()
    except Exception:
        save(ROOT/'error.json',{'traceback':traceback.format_exc()});raise
