"""Private Kaggle job; builds native GaussianCut then executes fixed scene presets."""
import argparse, gc, hashlib, inspect, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

ROOT=Path('/kaggle/working/experiment')
ROOT.mkdir(parents=True,exist_ok=True)
TMP=Path('/kaggle/tmp/remove360'); TMP.mkdir(parents=True,exist_ok=True)
os.environ['MAX_JOBS']='2';os.environ['TORCH_CUDA_ARCH_LIST']='7.5'
os.environ['OMP_NUM_THREADS']='2';os.environ['TOKENIZERS_PARALLELISM']='false'
URL='https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip'
REV='93d24a4ba02944ffae7848f9882b083cd9e04640'
report={'status':'started','scenes':[],'protocol':'Mip-NeRF360 3DGS removal pilot; pseudo-label evaluation, not Remove360 GT benchmark','started':time.time()}

def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2))
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def cmd(args,timeout=1800):
    print('RUN',args,flush=True);subprocess.run([str(x) for x in args],check=True,timeout=timeout)

def setup():
    import torch
    report['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}
    cmd([sys.executable,'-m','pip','install','--quiet','ninja','plyfile','PyMaxflow','remotezip','transformers==4.51.3','timm'])
    repo=TMP/'gaussiancut'
    cmd(['git','clone','--quiet','https://github.com/umangi-jain/gaussiancut.git',repo]);cmd(['git','-C',repo,'checkout','--quiet',REV])
    cmd(['git','-C',repo,'submodule','update','--init','--recursive'])
    cmd(['git','clone','--quiet','--depth','1','--branch','0.9.9.8','https://github.com/g-truc/glm.git',repo/'gaussian-splatting/submodules/diff-gaussian-rasterization/third_party/glm'])
    # Current compiler no longer supplies integer types through unrelated headers.
    # This adds the missing standard include, with no algorithm changes.
    header=repo/'gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h'
    original=header.read_text();header.write_text('#include <cstdint>\n'+original)
    patches=[{'file':str(header.relative_to(repo)),'change':'prepend #include <cstdint>','before_sha256':hashlib.sha256(original.encode()).hexdigest(),'after_sha256':digest(header)}]
    knn=repo/'gaussian-splatting/submodules/simple-knn/simple_knn.cu';original=knn.read_text();knn.write_text('#include <cfloat>\n'+original)
    patches.append({'file':str(knn.relative_to(repo)),'change':'prepend #include <cfloat> for FLT_MAX','before_sha256':hashlib.sha256(original.encode()).hexdigest(),'after_sha256':digest(knn)})
    save(ROOT/'compatibility_patches.json',{'glm_tag':'0.9.9.8','glm_commit':'bf71a834948186f4097caa076cd2663c69a10e1e','patches':patches})
    for name in ['diff-gaussian-rasterization','simple-knn']:
        with (ROOT/(name+'_build.log')).open('w') as f:
            p=subprocess.run([sys.executable,'-m','pip','install','-vv','--no-build-isolation',str(repo/'gaussian-splatting/submodules'/name)],stdout=f,stderr=subprocess.STDOUT,timeout=1200)
        if p.returncode:raise RuntimeError('Build failed: '+name+'; inspect persisted build log')
    sys.path.insert(0,str(repo/'gaussian-splatting'))
    cmd([sys.executable,'-m','pip','freeze'])
    return repo

def cameras(raw):
    import numpy as np
    from scene.cameras import Camera
    import torch
    selected=np.linspace(0,len(raw)-1,min(18,len(raw)),dtype=int).tolist()
    result=[]
    for j,k in enumerate(selected):
        r=raw[k];w=640;h=int(round(r['height']*w/r['width']))
        c2w=np.eye(4);c2w[:3,:3]=r['rotation'];c2w[:3,3]=r['position'];w2c=np.linalg.inv(c2w)
        fx=r['fx']*w/r['width'];fy=r['fy']*h/r['height']
        rec={'id':f'v{j:03d}','source_index':k,'source_image':r['img_name'],'image':f'v{j:03d}.png','width':w,'height':h,'fx':fx,'fy':fy,'cx':w/2,'cy':h/2,'world_to_camera':w2c.tolist()}
        cam=Camera(colmap_id=k,R=w2c[:3,:3].T,T=w2c[:3,3],FoVx=2*np.arctan(w/(2*fx)),FoVy=2*np.arctan(h/(2*fy)),image=torch.zeros(3,h,w),gt_alpha_mask=None,image_name=rec['id'],uid=j)
        result.append((rec,cam))
    return result

def model_tools():
    from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection,SamModel,SamProcessor
    from huggingface_hub import snapshot_download
    loc=snapshot_download('IDEA-Research/grounding-dino-tiny',allow_patterns=['*.json','*.safetensors','*.txt'])
    detproc=AutoProcessor.from_pretrained(loc);det=AutoModelForZeroShotObjectDetection.from_pretrained(loc).to('cuda').eval()
    samproc=SamProcessor.from_pretrained('facebook/sam-vit-base');sam=SamModel.from_pretrained('facebook/sam-vit-base').to('cuda').eval()
    return loc,detproc,det,samproc,sam

def detect(image,prompt,processor,model):
    import torch
    inputs=processor(images=image,text=prompt,return_tensors='pt').to('cuda')
    with torch.inference_mode():outputs=model(**inputs)
    fn=processor.post_process_grounded_object_detection
    key='threshold' if 'threshold' in inspect.signature(fn).parameters else 'box_threshold'
    result=fn(outputs,inputs.input_ids,**{key:.30},text_threshold=.25,target_sizes=[(image.height,image.width)])[0]
    return {'boxes':result['boxes'].cpu().tolist(),'scores':result['scores'].cpu().tolist()}

def segment(image,boxes,processor,model):
    import torch,numpy as np
    if not boxes:return np.zeros((image.height,image.width),bool)
    inputs=processor(images=image,input_boxes=[boxes],return_tensors='pt').to('cuda')
    with torch.inference_mode():out=model(**inputs)
    masks=processor.image_processor.post_process_masks(out.pred_masks.cpu(),inputs['original_sizes'].cpu(),inputs['reshaped_input_sizes'].cpu())[0]
    scores=out.iou_scores[0].cpu();best=scores.argmax(-1)
    return masks[range(len(boxes)),best].any(0).numpy().astype(bool)

