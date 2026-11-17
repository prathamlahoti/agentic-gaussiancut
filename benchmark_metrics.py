"""Remove360 evaluator bridge with explicit, tested paper/code compatibility fixes.

Semantic and depth scores call released code. Structural empty-set handling
follows paper G.4 (released implementation silently omits those views).
Depth uses a common linear 8-bit encoding per pair, reserving zero for invalid
pixels. Encoding is our adapter; it is not a claimed author-supplied renderer.
"""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor/remove360'))
from remove360.preprocessing.depth_diff import process_depth_pair
from remove360.evaluation.semantic import evaluate_semantic_scene,summarize_semantic_results
from remove360.evaluation.depth import evaluate_depth_scene,summarize_depth_results
from remove360.evaluation.sam import _calculate_iou_np,_resize_mask


def structural_similarity(before,after,target):
    def keep(masks):
        result=[]
        for mask in masks:
            m=_resize_mask(mask,target.shape[0],target.shape[1]).astype(bool)
            if _calculate_iou_np(m,target)>=.10:result.append(m)
        return result
    a,b=keep(before),keep(after)
    if not a or not b:
        return {'sim_sam':1. if not a and not b else 0.,'n_pre':len(a),'n_post':len(b),'paper_empty_set_fix':True}
    matrix=np.array([[_calculate_iou_np(x,y) for y in b] for x in a])
    rows,cols=linear_sum_assignment(-matrix)
    return {'sim_sam':float(matrix[rows,cols].sum()/max(len(a),len(b))),'n_pre':len(a),'n_post':len(b),'paper_empty_set_fix':False}


def depth_pair(before,after,out,ident):
    out=Path(out)
    valid=np.isfinite(before)&np.isfinite(after)&(before>0)&(after>0)
    if not valid.any():raise ValueError('No jointly valid depth pixels')
    lo=float(min(before[valid].min(),after[valid].min()))
    hi=float(max(before[valid].max(),after[valid].max()))
    for name,values in [('before',before),('after',after)]:
        p=out/'depth'/name;p.mkdir(parents=True,exist_ok=True)
        encoded=np.zeros(before.shape,np.uint8)
        encoded[valid]=np.clip(np.rint(1+254*(values[valid]-lo)/max(hi-lo,1e-8)),1,255).astype(np.uint8)
        Image.fromarray(encoded).save(p/(ident+'.png'))
    p=process_depth_pair(str(out/'depth/before'/(ident+'.png')),str(out/'depth/after'/(ident+'.png')),str(out/'depth_diff'),save_intermediate=True)
    # Float expected-depth exports can be NaN in transparent holes. Keep them
    # excluded; separately report target-valid fraction rather than call changed.
    return p,{'encoding':'joint linear min/max to [1,255], zero invalid','lo':lo,'hi':hi,'valid_fraction':float(valid.mean()),'save_intermediate':True}


def score_arm(folder):
    folder=Path(folder)
    semantic_rows=evaluate_semantic_scene(str(folder),verbose=False)
    semantic=summarize_semantic_results(semantic_rows,label='')
    detectable=[r for r in semantic_rows if r['iou_before']>=.30]
    semantic['detectable_pre_views']=len(detectable)
    semantic['acc_seg_30_detectable_only']=float(np.mean([r['iou_after']<.30 for r in detectable])) if detectable else None
    depth_rows=evaluate_depth_scene(str(folder),verbose=False)
    dep=summarize_depth_results(depth_rows,label='')
    rows=[]
    for target_file in sorted((folder/'masks').glob('*.png')):
        ident=target_file.stem;target=np.asarray(Image.open(target_file))>0
        a=np.load(folder/'sam/before'/(ident+'.npz'))['masks']
        b=np.load(folder/'sam/after'/(ident+'.npz'))['masks']
        row=structural_similarity(a,b,target);row['view']=ident;row['structural_change']=1-row['sim_sam']
        pre=np.asarray(Image.open(folder/'rgb/before'/(ident+'.png'))).astype(float)
        post=np.asarray(Image.open(folder/'rgb/after'/(ident+'.png'))).astype(float)
        diff=np.abs(pre-post).mean(-1)/255.
        row['rgb_mae_target']=float(diff[target].mean())
        row['rgb_mae_outside_target']=float(diff[~target].mean())
        rows.append(row)
    result={'semantic':semantic,'depth':dep,'structure':rows,'mean_structural_change':float(np.mean([x['structural_change'] for x in rows])),
            'protocol':'Remove360 Appendix I reference-free development subset',
            'physical_remove360_rqs':None,
            'limitations':['SAM-generated reference masks require visual review','Depth renderer and linear encoding are our declared adapter','Structural change is not physical correctness','GroundedSAM2 uses complete images, no target crop or target mask as prompt','Thresholds and test views fixed before evaluation']}
    eligibility=folder.parent/'references/eligibility.json'
    if eligibility.exists():
        quality=json.loads(eligibility.read_text());ids=set(quality['complete_views'])
        result['view_eligibility']=quality
        sr=[r for r in semantic_rows if Path(r['image_name']).stem in ids]
        dr=[r for r in depth_rows if r['image'] in ids]
        st=[r for r in rows if r['view'] in ids]
        result['complete_view_summary']={'n_views':len(sr),'semantic':summarize_semantic_results(sr,label=''),'depth':summarize_depth_results(dr,label=''),'structural_change':float(np.mean([r['structural_change'] for r in st])) if st else None}
        result['limitations'].append('Main pilot includes flagged partial target views; complete-view summary is reported separately, matching the paper coverage restriction more closely')
    (folder/'metrics.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args()
    print(json.dumps(score_arm(a.folder),indent=2))
