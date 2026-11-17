"""Executed after the pinned CUDA driver definitions. No evaluation feedback in policy.

Native GaussianCut defaults match Remove360 G.1. SAM and GroundingDINO are
released research implementations. Projection gates and action harness are ours.
"""
from PIL import Image, ImageDraw
import numpy as np


def overlay(image, mask):
    a = np.asarray(image).copy()
    a[mask] = (a[mask] * .55 + np.array([25,220,160]) * .45).astype('uint8')
    return Image.fromarray(a)


def sheet(tiles):
    canvas = Image.new('RGB', (800, 300*((len(tiles)+1)//2)), '#122b31')
    for i,(label,im) in enumerate(tiles):
        x,y = i%2*400,i//2*300
        im=im.copy(); im.thumbnail((390,266));canvas.paste(im,(x+5,y+30))
        ImageDraw.Draw(canvas).text((x+8,y+9),label,fill='white')
    return canvas


def choose(processor, model, image, prompt, labels):
    import torch
    chat=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
    inputs=processor(text=processor.apply_chat_template(chat,add_generation_prompt=True),images=[image],return_tensors='pt').to('cuda')
    ids=[processor.tokenizer.encode(x,add_special_tokens=False) for x in labels]
    assert all(len(x)==1 for x in ids)
    with torch.inference_mode(): p=model(**inputs).logits[0,-1,[x[0] for x in ids]].float().softmax(0).cpu().numpy()
    return labels[int(p.argmax())],dict(zip(labels,map(float,p)))


def backproject(rec, mask, depth, limit=6000):
    y,x=np.nonzero(mask & np.isfinite(depth) & (depth>0))
    if len(x)>limit:
        ix=np.linspace(0,len(x)-1,limit,dtype=int);x=x[ix];y=y[ix]
    z=depth[y,x]
    p=np.stack([(x-rec['cx'])*z/rec['fx'],(y-rec['cy'])*z/rec['fy'],z,np.ones_like(z)],1)
    return p @ np.linalg.inv(np.asarray(rec['world_to_camera'])).T


def projection(points, rec, depth):
    p=points @ np.asarray(rec['world_to_camera']).T
    uv=p[:,:2]/np.maximum(p[:,2:3],1e-8)*[rec['fx'],rec['fy']]+[rec['cx'],rec['cy']]
    good=(p[:,2]>.01)&(uv[:,0]>=2)&(uv[:,0]<rec['width']-2)&(uv[:,1]>=2)&(uv[:,1]<rec['height']-2)
    inside=float(good.mean()) if len(good) else 0.
    uv=uv[good]; z=p[good,2]
    pix=np.rint(uv).astype(int)
    if len(pix)==0:return None
    obs=depth[pix[:,1],pix[:,0]]
    visible=np.isfinite(obs)&(np.abs(obs-z)<np.maximum(.03,.06*z))
    pix=pix[visible]
    if len(pix)<24:return None
    lo,hi=np.quantile(pix,[.01,.99],axis=0)
    pad=np.maximum((hi-lo)*.12,6)
    box=np.r_[np.maximum(lo-pad,0),np.minimum(hi+pad,[rec['width']-1,rec['height']-1])]
    return {'pixels':pix,'box':box.tolist(),'inside':inside,'visible':float(visible.mean()),'area':float(np.prod(hi-lo))}


def mask_quality(mask, proj, rec, depth, seed_rec, seed_mask, seed_depth):
    pix=proj['pixels']; coverage=float(mask[pix[:,1],pix[:,0]].mean())
    rev=projection(backproject(rec,mask,depth),seed_rec,seed_depth)
    agreement=0.
    if rev is not None:
        xy=rev['pixels'];agreement=float(seed_mask[xy[:,1],xy[:,0]].mean())
    ys,xs=np.nonzero(mask)
    clipped=bool(len(xs) and ((xs.min()<2) or (ys.min()<2) or (xs.max()>=mask.shape[1]-2) or (ys.max()>=mask.shape[0]-2)))
    return {'seed_coverage':coverage,'reverse_agreement':agreement,'mask_fraction':float(mask.mean()),'clipped':clipped,
            'accepted':bool(coverage>=.60 and agreement>=.65 and .001<mask.mean()<.45 and not clipped)}


def sam_mask(image, proj, predictor, detector, detproc, prompt):
    # GroundingDINO proposes complete object boxes; projection selects the instance.
    d=detect(image,prompt,detproc,detector)
    pix=proj['pixels'];boxes=[]
    for box,score in zip(d['boxes'],d['scores']):
        x0,y0,x1,y1=box
        fraction=((pix[:,0]>=x0)&(pix[:,0]<=x1)&(pix[:,1]>=y0)&(pix[:,1]<=y1)).mean()
        if fraction>=.65:boxes.append((float(fraction)+score,box))
    boxes=sorted(boxes,reverse=True)[:2]
    boxes.append((0,proj['box']))
    predictor.set_image(np.asarray(image))
    # Interior projected positive points constrain SAM away from floor/sofa.
    centroid=np.median(pix,axis=0)
    order=np.argsort(np.linalg.norm(pix-centroid,axis=1))
    pts=pix[order[np.linspace(0,max(0,len(order)//2-1),3,dtype=int)]].astype(float)
    proposals=[]
    for _,box in boxes:
        masks,scores,_=predictor.predict(point_coords=pts,point_labels=np.ones(len(pts),int),box=np.array(box),multimask_output=True)
        for mask,score in zip(masks,scores):proposals.append((mask,float(score),box))
    return proposals,d


def render_scene(name, repo):
    import torch
    from remotezip import RemoteZip
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from types import SimpleNamespace as NS
    folder=ROOT/'v2'/'scenes'/name;folder.mkdir(parents=True,exist_ok=True)
    checkpoint=TMP/(name+'.ply')
    with RemoteZip(URL) as z:
        raw=json.loads(z.read(name+'/cameras.json'))
        with z.open(name+'/point_cloud/iteration_30000/point_cloud.ply') as src,checkpoint.open('wb') as dst:shutil.copyfileobj(src,dst)
    pc=GaussianModel(3);pc.load_ply(str(checkpoint));pipe=NS(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda');records={};images={};depths={}
    with torch.inference_mode():
        xyz=torch.cat([pc.get_xyz,torch.ones_like(pc.get_xyz[:,:1])],1)
        for rec,cam in cameras(raw):
            rgb=render(cam,pc,pipe,bg)['render'].clamp(0,1)
            im=Image.fromarray((rgb.permute(1,2,0).cpu().numpy()*255).round().astype('uint8'))
            z=(xyz @ cam.world_view_transform)[:,2]
            col=z[:,None].expand(-1,3).contiguous()
            n=render(cam,pc,pipe,bg,override_color=col)['render'][0]
            a=render(cam,pc,pipe,bg,override_color=torch.ones_like(col))['render'][0]
            dep=torch.where(a>1e-5,n/a.clamp_min(1e-5),torch.nan).cpu().numpy()
            ident=rec['id'];records[ident]=rec;images[ident]=im;depths[ident]=dep
            im.save(folder/rec['image']);np.save(folder/(ident+'.npy'),dep)
    save(folder/'cameras.json',records)
    save(folder/'provenance.json',{'source':URL,'member':name+'/point_cloud/iteration_30000/point_cloud.ply','sha256':digest(checkpoint)})
    del pc,xyz,z,col,n,a,rgb,cam;gc.collect();torch.cuda.empty_cache()
    return checkpoint,folder,records,images,depths


def run_case(cfg, case, scene, repo, models):
    import torch
    checkpoint,frames,recs,images,depths=scene
    proc,vlm,predictor,detproc,det=models
    out=ROOT/'v2'/case['id'];out.mkdir(parents=True,exist_ok=True)
    seed=case['seed'];seed_im=images[seed]
    predictor.model.to('cuda');det.to('cuda');vlm.to('cuda')
    if 'polygon' in case:
        m=Image.new('L',seed_im.size);ImageDraw.Draw(m).polygon([tuple(x) for x in case['polygon']],fill=255);seed_mask=np.asarray(m)>0
        seed_origin='one provided hand-drawn mask, unchanged from v1'
    else:
        predictor.set_image(np.asarray(seed_im))
        masks,scores,_=predictor.predict(box=np.array(case['seed_box']),multimask_output=True)
        seed_mask=masks[int(scores.argmax())]
        seed_origin='SAM from one manually specified seed box; inspect before interpreting as supplied mask'
    Image.fromarray(seed_mask.astype('uint8')*255).save(out/'seed_mask.png')
    overlay(seed_im,seed_mask).save(out/'seed_overlay.png')
    points=backproject(recs[seed],seed_mask,depths[seed])
    pool={}
    for ident in recs:
        if ident==seed or ident in case['held_out']:continue
        p=projection(points,recs[ident],depths[ident])
        if p and p['inside']>=.90 and p['visible']>=.35 and p['area']>=150:pool[ident]=p
    save(out/'protocol.json',{'config':case,'seed_origin':seed_origin,'pool':{k:{a:b for a,b in v.items() if a!='pixels'} for k,v in pool.items()},'request_budget':cfg['max_requests'],'held_out_excluded':True,'thumbnail_previews':True})
    results=[]
    for policy in ['fixed','vlm']:
        arm=out/policy;arm.mkdir(exist_ok=True);maskdir=arm/'masks';maskdir.mkdir(exist_ok=True)
        shutil.copy2(out/'seed_mask.png',maskdir/(seed+'.png'))
        accepted=[seed];remaining=sorted(pool);trace=[]
        for step in range(cfg['max_requests']):
            if not remaining:break
            if policy=='fixed': chosen=remaining[0];action={'policy':'fixed ascending camera ID'}
            else:
                # Geometry screens unusable projections, VLM picks among all remaining previews.
                labels=[chr(65+i) for i in range(len(remaining))]
                mapping=dict(zip(labels,remaining));tiles=[('TARGET: initial mask',overlay(seed_im,seed_mask))]
                for ident in accepted[1:]:
                    tiles.append(('ACCEPTED '+ident,Image.open(arm/(ident+'_overlay.png')).convert('RGB')))
                for label,ident in mapping.items():
                    im=images[ident].copy();ImageDraw.Draw(im).rectangle(pool[ident]['box'],outline='#24d8ab',width=3)
                    tiles.append((label+' '+ident,im))
                rejected=[x['view'] for x in trace if x.get('quality',{}).get('accepted') is False]
                prompt=f'Select a view to segment the SAME {case["target"]}. The first panel marks the target. ACCEPTED panels show the masks collected so far. Green boxes on candidate panels are approximate projections, not masks. Prefer a complete visible boundary and a different side from accepted views {accepted}. Rejected masks so far: {rejected}. Avoid clutter, occlusion and image edges. Choose one letter: '+', '.join(labels)+'.'
                if len(accepted)>=3:
                    mapping['S']='stop';labels.append('S');prompt+=' S means enough distinct complete views have already been accepted; stop.'
                im=sheet(tiles);im.save(arm/f'decision_{step+1}.jpg')
                letter,scores=choose(proc,vlm,im,prompt,labels);chosen=mapping[letter]
                action={'policy':'VLM letter action','scores':scores,'options':mapping,'prompt':prompt}
            action.update(step=step+1,view=chosen)
            if chosen=='stop':trace.append(action);break
            remaining.remove(chosen)
            proposals,detection=sam_mask(images[chosen],pool[chosen],predictor,det,detproc,case['prompt'])
            rated=[]
            for mask,score,box in proposals:
                quality=mask_quality(mask,pool[chosen],recs[chosen],depths[chosen],recs[seed],seed_mask,depths[seed])
                rated.append((quality['accepted'],quality['seed_coverage']+quality['reverse_agreement']+score,mask,quality,box,score))
            best=max(rated,key=lambda x:(x[0],x[1]));_,_,mask,quality,box,score=best
            action.update(quality=quality,sam_predicted_iou=score,sam_box=box,detector=detection)
            overlay(images[chosen],mask).save(arm/(chosen+'_overlay.png'))
            if quality['accepted']:
                Image.fromarray(mask.astype('uint8')*255).save(maskdir/(chosen+'.png'));accepted.append(chosen)
            trace.append(action);save(arm/'trace.json',trace);print('V2_ACTION',case['id'],policy,json.dumps(action),flush=True)
        save(arm/'trace.json',trace)
        stop='vlm_stop' if trace and trace[-1]['view']=='stop' else 'budget_or_pool_exhausted'
        save(arm/'selection.json',{'accepted':accepted,'stop':stop,'sam_requests':sum(x['view']!='stop' for x in trace)})
        if len(accepted)<2:
            results.append({'policy':policy,'status':'insufficient_valid_masks','accepted':accepted});continue
        selected=[dict(recs[i],split='train' if i in accepted else 'eval') for i in accepted+case['held_out']]
        save(arm/'cameras.json',selected)
        run=ROOT/'removal_pipeline/runs'/(case['id']+'_'+policy);run.mkdir(parents=True,exist_ok=True)
        conf={'output':str(run),'upstream':str(repo),'checkpoint':str(checkpoint),'sh_degree':3,'cameras':str(arm/'cameras.json'),'rgb_frames':str(frames),'masks':str(maskdir),'target_identifier':case['id'],'foreground_threshold':.9,'seed':cfg['seed']}
        save(arm/'config.json',conf)
        for model in [vlm,predictor.model,det]:model.to('cpu')
        gc.collect();torch.cuda.empty_cache()
        with (arm/'removal.log').open('w') as f:
            p=subprocess.run([sys.executable,str(ROOT/'removal_pipeline/gpu_adapter.py'),'--config',str(arm/'config.json')],stdout=f,stderr=subprocess.STDOUT,timeout=2700)
        if p.returncode:raise RuntimeError((arm/'removal.log').read_text()[-3000:])
        manifest=json.loads((run/'manifest.json').read_text())
        # Remove duplicate intermediate PLYs only; retain final edited PLY in output.
        for path in run.glob('graphcut_*/*.ply'):path.unlink()
        results.append({'policy':policy,'status':'edited','accepted':accepted,'run':str(run),'removed':manifest['gaussians_before']-manifest['gaussians_after'],'stop':stop})
        save(out/'results.json',results)
        for model in [vlm,predictor.model,det]:model.to('cuda')
    return results


def main():
    import torch
    cfg=json.loads((ROOT/'v2_config.json').read_text())
    np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    all_results=[]
    try:
        repo=setup()
        cmd([sys.executable,'-m','pip','install','--quiet','git+https://github.com/facebookresearch/segment-anything.git@'+cfg['sam_revision']])
        from segment_anything import sam_model_registry,SamPredictor
        from transformers import AutoProcessor,AutoModelForVision2Seq,AutoModelForZeroShotObjectDetection
        weights=TMP/'sam_vit_h_4b8939.pth'
        import urllib.request
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',weights)
        cache={}
        for name in dict.fromkeys(c['scene'] for c in cfg['cases']):cache[name]=render_scene(name,repo)
        proc=AutoProcessor.from_pretrained(cfg['vlm']);vlm=AutoModelForVision2Seq.from_pretrained(cfg['vlm'],torch_dtype=torch.float16).to('cuda').eval()
        predictor=SamPredictor(sam_model_registry['vit_h'](checkpoint=str(weights)).to('cuda').eval())
        dp=AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny');det=AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny').to('cuda').eval()
        models=(proc,vlm,predictor,dp,det)
        for case in cfg['cases']:
            try:r=run_case(cfg,case,cache[case['scene']],repo,models);all_results.append({'case':case['id'],'results':r})
            except Exception:
                traceback.print_exc();all_results.append({'case':case['id'],'error':traceback.format_exc()})
            save(ROOT/'v2/report.json',all_results)
    except Exception:
        save(ROOT/'v2/fatal.json',{'error':traceback.format_exc()});raise
    print('V2_COMPLETE',json.dumps(all_results),flush=True)


if __name__=='__main__':main()
