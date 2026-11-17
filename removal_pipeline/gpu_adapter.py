"""Untested on GPU here. Calls pinned upstream GaussianCut without changing its algorithm.

The upstream graphcut return mask marks BACKGROUND with 1. Keep that subset
for removal; upstream gaussians_source.ply is the extracted TARGET.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace as NS

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    c=json.loads(Path(a.config).read_text());out=Path(c['output'])
    from runner import writable
    writable(out)
    sys.path.insert(0,str(Path(c['upstream'])/'gaussian-splatting'))
    import numpy as np
    import torch
    from PIL import Image
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.render_utils import render_coarse_sets
    from utils.graphcut import graphcut_segmentation, load_gc_args
    from runner import digest

    random.seed(c.get('seed',0));np.random.seed(c.get('seed',0));torch.manual_seed(c.get('seed',0));torch.cuda.manual_seed_all(c.get('seed',0))
    timings={}; start=time.perf_counter()
    records=json.loads(Path(c['cameras']).read_text());cams=[]
    for i,r in enumerate(records):
        with Image.open(Path(c['rgb_frames'])/r['image']) as im:
            image=torch.from_numpy(np.asarray(im.convert('RGB')).copy()).permute(2,0,1).float()/255
        m=np.asarray(r['world_to_camera'],dtype=np.float32)
        cams.append(Camera(colmap_id=i,R=m[:3,:3].T,T=m[:3,3],FoVx=2*np.arctan(r['width']/(2*r['fx'])),FoVy=2*np.arctan(r['height']/(2*r['fy'])),image=image,gt_alpha_mask=None,image_name=r['id'],uid=i))
    model=GaussianModel(c['sh_degree']); model.load_ply(c['checkpoint'])
    before_count=len(model._xyz)
    if before_count<10: raise ValueError('At least 10 Gaussians required for default kNN.')
    pipeline=NS(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    train=[cam for cam,r in zip(cams,records) if r['split']=='train']; evaluation=[cam for cam,r in zip(cams,records) if r['split']=='eval']
    scene=NS(getTrainCameras=lambda:train,getTestCameras=lambda:evaluation)
    # Stage only training masks so held-out scoring masks cannot influence editing.
    maskdir=out/'inputs'/'multiview_masks'; maskdir.mkdir(parents=True)
    for r in records:
        mask=Path(c['masks'])/(r['id']+'.png')
        if r['split']=='train' and mask.is_file(): shutil.copy2(mask,maskdir/mask.name)
    dataset=NS(model_path=str(out),source_path=c['rgb_frames'],white_background=False)
    (out/('graphcut_'+c['target_identifier'])).mkdir()
    torch.cuda.synchronize(); t=time.perf_counter()
    weights_source,weights_sink,source,sink=render_coarse_sets(model,scene,str(out/'inputs'),dataset,pipeline,'multiview',c['foreground_threshold'],c['target_identifier'],[])
    torch.cuda.synchronize();timings['coarse_seconds']=time.perf_counter()-t
    if min(len(source._xyz),len(sink._xyz))<5: raise ValueError('Foreground/background need at least five Gaussians for terminal clusters.')
    p=argparse.ArgumentParser();load_gc_args(p);args=p.parse_args([]);args.identifier=c['target_identifier']
    gc=NS(num_edges=10,terminal_clusters_source=5,terminal_clusters_sink=5,leaf_size=40)
    t=time.perf_counter()
    _,background_index=graphcut_segmentation(args,str(out),gc,weights_source,weights_sink,model,source,sink)
    torch.cuda.synchronize();timings['graphcut_seconds']=time.perf_counter()-t
    keep=torch.as_tensor(background_index==1,device=model._xyz.device)
    if not keep.any() or keep.all(): raise ValueError('Degenerate cut: all or no Gaussians removed; inspect masks and graph.')
    edited=copy.deepcopy(model);edited.remove_low_score_gaussians(keep)
    edited.save_ply(str(out/'edited_background.ply'))

    black=torch.zeros(3,device='cuda')
    def export(pc,cam,arm):
        rgb=render(cam,pc,pipeline,black)['render'].detach().clamp(0,1)
        # Same rasterizer/visibility/opacity as RGB. Weighted expected camera-z
        # is sum(T*alpha*z)/sum(T*alpha), using center depths and black BG.
        xyz=torch.cat((pc.get_xyz,torch.ones_like(pc.get_xyz[:,:1])),dim=1)
        z=(xyz@cam.world_view_transform)[:,2]
        colors=z[:,None].expand(-1,3).contiguous()
        numerator=render(cam,pc,pipeline,black,override_color=colors)['render'][0]
        alpha=render(cam,pc,pipeline,black,override_color=torch.ones_like(colors))['render'][0]
        valid=(alpha>1e-5)&torch.isfinite(numerator)
        depth=torch.where(valid,numerator/alpha.clamp_min(1e-5),torch.nan)
        if not torch.isfinite(rgb).all(): raise ValueError('Nonfinite RGB')
        for sub in ('rgb','depth','alpha'):(out/sub/arm).mkdir(parents=True,exist_ok=True)
        Image.fromarray((rgb.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(out/'rgb'/arm/(cam.image_name+'.png'))
        np.save(out/'depth'/arm/(cam.image_name+'.npy'),depth.detach().cpu().numpy().astype(np.float32))
        np.save(out/'alpha'/arm/(cam.image_name+'.npy'),alpha.detach().cpu().numpy().astype(np.float32))
    t=time.perf_counter()
    with torch.no_grad():
        for cam in cams:
            export(model,cam,'before');export(edited,cam,'after')
    torch.cuda.synchronize();timings['paired_render_seconds']=time.perf_counter()-t
    timings['total_seconds']=time.perf_counter()-start
    shutil.copy2(c['cameras'],out/'cameras.json')
    outputs=[]
    for r in records:
        for arm in ('before','after'):
            for kind,ext in [('rgb','png'),('depth','npy'),('alpha','npy')]:
                path=out/kind/arm/(r['id']+'.'+ext)
                if not path.is_file(): raise ValueError('Missing paired output: '+str(path))
                outputs.append({'path':str(path.relative_to(out)),'sha256':digest(path)})
    lock=json.loads((Path(__file__).parent/'upstream_lock.json').read_text())
    manifest={'status':'completed_3dgs_removal_pending_scientific_review','completed_3d_removal':True,'baseline':'GaussianCut removal-only; background partition retained; no inpainting','upstream_revision':lock['gaussiancut']['revision'],'settings':c,'graphcut_defaults':vars(args),'graph_parameters':vars(gc),'camera_sha256':digest(out/'cameras.json'),'cameras_identical_by_construction':True,'checkpoint_before_sha256':digest(c['checkpoint']),'checkpoint_after':'edited_background.ply','checkpoint_after_sha256':digest(out/'edited_background.ply'),'gaussians_before':before_count,'gaussians_after':len(edited._xyz),'depth_definition':'alpha-normalized expected camera-space Gaussian-center z, scene units; NaN at alpha<=1e-5; same rasterizer as RGB','depth_is_official_remove360_export':False,'timings':timings,'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},'outputs':outputs,'residual_attack_evaluated':False,'rqs':None}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
