"""CPU preflight + explicit execution entry point. All writes stay inside this directory."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def writable(p):
    p = p.resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('Writable path must stay inside removal_pipeline: ' + str(p))
    return p

def load_config(path):
    path = Path(path).resolve()
    c = json.loads(path.read_text())
    for k in ('checkpoint', 'cameras', 'rgb_frames', 'masks', 'upstream', 'output'):
        c[k] = str((path.parent / c[k]).resolve())
    writable(Path(c['output']))
    writable(Path(c['upstream']))
    return c

def check(c):
    errors, observations, inputs = [], {}, {}
    for k in ('checkpoint', 'cameras', 'rgb_frames', 'masks'):
        p = Path(c[k])
        if not p.exists(): errors.append('Missing ' + k + ': ' + str(p))
        elif p.is_file(): inputs[k] = {'path': str(p), 'sha256': digest(p)}
    if not re.fullmatch(r'[A-Za-z0-9_-]+', c['target_identifier']) or c['target_identifier'] == 'SELECT_A_TARGET':
        errors.append('Select a concrete target_identifier (letters, digits, underscore, hyphen).')
    if c.get('sh_degree') not in (0, 1, 2, 3): errors.append('sh_degree must be 0..3 and match PLY.')
    if not 0 < c.get('foreground_threshold', 0) < 1: errors.append('foreground_threshold must lie strictly between zero and one.')
    if Path(c['checkpoint']).is_file():
        with open(c['checkpoint'], 'rb') as f:
            header = []
            for _ in range(300):
                line = f.readline().decode('ascii', errors='replace').strip(); header.append(line)
                if line == 'end_header': break
        props = [x.split()[-1] for x in header if x.startswith('property ')]
        needed = ['x','y','z','opacity'] + ['f_dc_'+str(i) for i in range(3)] + ['scale_'+str(i) for i in range(3)] + ['rot_'+str(i) for i in range(4)]
        if header[0] != 'ply' or any(x not in props for x in needed): errors.append('Checkpoint is not a compatible Graphdeco Gaussian PLY.')
        expected = 3 * ((c.get('sh_degree',3)+1)**2-1)
        if len([x for x in props if x.startswith('f_rest_')]) != expected: errors.append('PLY spherical harmonic degree does not match config.')
    if Path(c['cameras']).is_file():
        try:
            import numpy as np
            from PIL import Image
            cams = json.loads(Path(c['cameras']).read_text())
            assert isinstance(cams, list) and cams, 'nonempty camera list required'
            ids=set(); train_masks=0; frame_inputs=[]
            for cam in cams:
                name=cam['id']; assert re.fullmatch(r'[A-Za-z0-9_-]+',name), 'unsafe camera id'
                assert name not in ids, 'duplicate camera id'; ids.add(name)
                assert cam['split'] in ('train','eval'), 'split must be train/eval'
                w,h=cam['width'],cam['height']; assert w>0 and h>0 and cam['fx']>0 and cam['fy']>0
                assert abs(cam['cx']-w/2)<1e-5 and abs(cam['cy']-h/2)<1e-5, 'only centered pinhole intrinsics supported; undistort/recenter first'
                m=np.asarray(cam['world_to_camera'],dtype=float)
                assert m.shape==(4,4) and np.isfinite(m).all() and np.allclose(m[3],[0,0,0,1])
                assert np.allclose(m[:3,:3]@m[:3,:3].T,np.eye(3),atol=1e-4) and abs(np.linalg.det(m[:3,:3])-1)<1e-4, 'pose must be rigid w2c'
                p=(Path(c['rgb_frames'])/cam['image']).resolve(); assert p.is_relative_to(Path(c['rgb_frames'])), 'frame outside RGB root'
                with Image.open(p) as im: assert im.size==(w,h), 'RGB dimensions mismatch'
                frame_inputs.append({'id':name,'rgb_sha256':digest(p)})
                mask=Path(c['masks'])/(name+'.png')
                if mask.is_file():
                    with Image.open(mask) as im:
                        a=np.asarray(im); assert im.size==(w,h) and a.ndim==2, 'mask must be single-channel and match RGB'
                        assert set(np.unique(a)).issubset({0,1,255}), 'mask must be binary'
                        if cam['split']=='train':
                            assert (a>0).any() and (a==0).any(), 'training mask must have foreground and background'
                            train_masks+=1
                    frame_inputs[-1]['mask_sha256']=digest(mask)
            assert train_masks>0, 'At least one training-view target mask required'
            assert any(x['split']=='eval' for x in cams), 'At least one held-out evaluation camera required'
            inputs['frames']=frame_inputs
            observations['camera_count']=len(cams)
        except Exception as e: errors.append('Camera/RGB/mask contract invalid: '+str(e))
    upstream=Path(c['upstream'])
    lock=json.loads((ROOT/'upstream_lock.json').read_text())['gaussiancut']
    try:
        sha=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
        observations['upstream_revision']=sha
        if sha!=lock['revision']: errors.append('Upstream revision mismatch')
        dirty=subprocess.check_output(['git','-C',str(upstream),'status','--porcelain','--untracked-files=no'],text=True)
        if dirty.strip(): errors.append('Upstream tracked files modified; use clean pinned checkout')
    except Exception: errors.append('Pinned GaussianCut checkout missing; see SETUP.md')
    for mod in ('numpy','PIL','torch','torchvision','plyfile','scipy','sklearn','maxflow','diff_gaussian_rasterization','simple_knn','cv2','tqdm'):
        try: found=importlib.util.find_spec(mod) is not None
        except Exception: found=False
        if not found: errors.append('Python dependency missing: '+mod)
    if importlib.util.find_spec('diff_gaussian_rasterization'):
        try:
            from diff_gaussian_rasterization import GaussianRasterizer
            if not hasattr(GaussianRasterizer, 'apply_weights'):
                errors.append('Wrong rasterizer: GaussianCut custom apply_weights is missing; install its bundled extension.')
        except Exception as e: errors.append('Rasterizer import failed: '+str(e))
    try:
        import torch
        observations.update(python=sys.version,torch=torch.__version__,cuda_available=torch.cuda.is_available(),cuda_build=torch.version.cuda)
        if not torch.cuda.is_available(): errors.append('CUDA GPU unavailable in this Python environment')
        else: observations['gpu']=torch.cuda.get_device_name(0)
    except Exception as e: errors.append('Torch inspection failed: '+str(e))
    return {'status':'blocked' if errors else 'preflight_passed','completed_3d_removal':False,'blockers':errors,'observations':observations,'input_hashes':inputs,'settings':c,'upstream':lock['revision']}

def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['preflight','run']);p.add_argument('--config',default=str(ROOT/'config.example.json'));p.add_argument('--report',default=str(ROOT/'preflight.json'));a=p.parse_args()
    c=load_config(a.config); report=check(c); dest=writable(Path(a.report));dest.parent.mkdir(parents=True,exist_ok=True)
    if a.mode=='run' and not report['blockers']:
        out=writable(Path(c['output']))
        out.mkdir(parents=True,exist_ok=False)
        config=out/'resolved_config.json';config.write_text(json.dumps(c,indent=2))
        cmd=[sys.executable,str(ROOT/'gpu_adapter.py'),'--config',str(config)]
        report['command']=cmd; t=time.perf_counter()
        with open(out/'execution.log','w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        report['elapsed_seconds']=time.perf_counter()-t;report['returncode']=result.returncode
        report['status']='adapter_finished_verify_outputs' if result.returncode==0 else 'execution_failed'
        # GPU adapter emits its own manifest only after validating paired exports.
        if result.returncode==0:
            report['result_manifest']=str(out/'manifest.json')
            manifest=json.loads((out/'manifest.json').read_text())
            report['completed_3d_removal']=manifest['completed_3d_removal']
        (out/'runner_report.json').write_text(json.dumps(report,indent=2))
    dest.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    return 2 if report['blockers'] or report.get('returncode',0)!=0 else 0

if __name__=='__main__': sys.exit(main())
