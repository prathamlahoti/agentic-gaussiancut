"""Build a meeting report from real run records. No invented scores or images."""
import argparse,base64,html,json,hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def esc(s):return html.escape(str(s))
def img(p,alt):
    p=Path(p)
    if not p.is_file():return '<p>Image unavailable</p>'
    mime='image/png' if p.suffix=='.png' else 'image/jpeg'
    return f'<img alt="{esc(alt)}" src="data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}">'
def pct(x):return 'pending' if x is None else f'{x*100:.1f}%'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=HERE/'v2_output/experiment');p.add_argument('--evaluation',type=Path,default=HERE/'evaluation_output/evaluation');a=p.parse_args()
    cfg=read(a.run/'v2_config.json');report=read(a.run/'v2/report.json')
    metrics=read(a.evaluation/'summary.json') if (a.evaluation/'summary.json').exists() else []
    indexed={(r['case'],r['policy']):r['metrics'] for r in metrics}
    css='''body{margin:0;background:#f5f7f4;color:#172c32;font:17px/1.55 Arial,sans-serif}main{max-width:1200px;margin:auto;padding:45px 30px}h1{font-size:44px;line-height:1.1;margin:0 0 18px}h2{font-size:30px;border-top:1px solid #c8d5ce;padding-top:32px;margin-top:45px}h3{font-size:22px}p{max-width:1000px}a{color:#146a57}nav{display:flex;gap:20px;flex-wrap:wrap;margin:25px 0}img{max-width:100%;height:auto;display:block}figure{margin:0}figcaption{font-size:15px;color:#445b62;margin:8px 0 20px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}.triple{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.notice{border-left:4px solid #b16c1d;padding:4px 18px;background:#f5ecd9}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:12px 10px;text-align:left;border-bottom:1px solid #d2dbd4}th{background:#e9efea}details{margin:20px 0}summary{cursor:pointer;font-weight:bold}code{font-size:14px}footer{margin-top:40px;font-size:14px;color:#445b62}.scroll{overflow-x:auto}@media(max-width:760px){h1{font-size:32px}.pair,.triple{grid-template-columns:1fr}main{padding:24px 18px}}'''
    blocks=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Agentic GaussianCut: experiment report</title><style>'+css+'</style><main>',
        '<h1>Agent-guided object removal from 3DGS</h1><p>One initial object mask. A VLM requests additional views. SAM segments the target, the harness checks each mask, and GaussianCut edits the 3D scene.</p>',
        '<p class="notice">Development study on three targets in two Mip-NeRF360 scenes. Evaluation follows Remove360’s reference-free setting. These are not full physical Remove360 benchmark scores. Images below are actual renders.</p>',
        '<nav>'+''.join(f'<a href="#{esc(c["id"])}">{esc(c["target"].split(" including")[0])}</a>' for c in cfg['cases'])+'</nav>',
        '<h2>What changed since the first pilot</h2><p>The first harness accepted masks based on size. The revised harness also checks projected target coverage, agreement with the initial mask from the original camera, and frame clipping. SAM ViT-H uses target points and object-box proposals. The VLM sees accepted masks before deciding what to request next.</p>',
        '<p>The fixed-order baseline and VLM use the same tools and a maximum of four additional requests. Candidate thumbnails are already rendered. Evaluation views are held out from editing, not from training the original splat.</p>',
        '<h2>Measured results</h2><p>IoU measures overlap of the full-image object detector’s mask with the target reference. A low post-edit IoU is useful only when the object was detectable before editing. Depth and structural scores measure change, not the correctness of the exposed background.</p>',
        '<div class="scroll"><table><thead><tr><th>Target / policy</th><th>Pre IoU</th><th>Post IoU</th><th>Non-detection @0.3</th><th>Depth change</th><th>Structural change</th></tr></thead><tbody>']
    rows=[]
    for case in cfg['cases']:
        for policy in ['original_v1','fixed','vlm']:
            m=indexed.get((case['id'],policy))
            if not m:continue
            s=m['semantic'];values=[s['mean_iou_before'],s['mean_iou_after'],s['acc_seg_30'],m['depth']['mean_acc_delta_depth'],m['mean_structural_change']]
            blocks.append('<tr><td>'+esc(case['id']+' / '+policy)+'</td>'+''.join('<td>'+pct(x)+'</td>' for x in values)+'</tr>')
            rows.append({'case':case['id'],'policy':policy,'values':values})
    if not rows:blocks.append('<tr><td colspan="6">Evaluation has not completed. No new benchmark score is claimed.</td></tr>')
    blocks.append('</tbody></table></div><p>Reference masks are SAM-generated and require visual review. Depth uses a declared common-scale encoding. See the protocol for the two tested paper/code compatibility fixes.</p>')
    deck_cases=[]
    for case in cfg['cases']:
        src=a.run/'v2'/case['id'];blocks.append(f'<h2 id="{esc(case["id"])}">{esc(case["id"].replace("_"," ").title())}</h2>')
        if not (src/'protocol.json').exists():
            err=next((x for x in report if x['case']==case['id']),{})
            blocks.append('<p>Case did not complete.</p><pre>'+esc(err.get('error','No result'))+'</pre>');continue
        protocol=read(src/'protocol.json')
        blocks.append('<div class="pair"><figure>'+img(src/'seed_overlay.png','Initial target mask')+'<figcaption>Input: initial target mask</figcaption></figure><div><h3>Input and constraints</h3><p>'+esc(protocol['seed_origin'])+'</p><p>Target: '+esc(case['target'])+'</p><p>Maximum additional requests: '+str(cfg['max_requests'])+'. Held-out cameras: '+esc(', '.join(case['held_out']))+'.</p></div></div>')
        for policy in ['fixed','vlm']:
            arm=src/policy
            if not (arm/'selection.json').exists():continue
            sel=read(arm/'selection.json');trace=read(arm/'trace.json')
            blocks.append('<h3>'+('VLM decisions' if policy=='vlm' else 'Fixed-order baseline')+'</h3><p>'+esc(str(sel['sam_requests'])+' SAM requests; accepted masks: '+', '.join(sel['accepted'])+'. End condition: '+sel['stop'])+'</p><div class="triple">')
            for step in trace:
                if step['view']=='stop':continue
                q=step['quality'];label=step['view']+' — '+('accepted' if q['accepted'] else 'rejected')
                blocks.append('<figure>'+img(arm/(step['view']+'_overlay.png'),label)+'<figcaption>'+esc(label)+'<br>Target-point coverage '+pct(q['seed_coverage'])+'; reverse agreement '+pct(q['reverse_agreement'])+'.</figcaption></figure>')
            blocks.append('</div>')
            if policy=='vlm':
                blocks.append('<details><summary>Exactly what the VLM saw</summary>')
                for f in sorted(arm.glob('decision_*.jpg')):blocks.append(img(f,f.stem))
                blocks.append('</details>')
            run=a.run/'removal_pipeline/runs'/(case['id']+'_'+policy)
            if not (run/'manifest.json').exists():blocks.append('<p>No edited output: too few valid masks or execution failure.</p>');continue
            manifest=read(run/'manifest.json')
            blocks.append('<p>Gaussians removed: '+f'{manifest["gaussians_before"]-manifest["gaussians_after"]:,}'+'. This count alone does not establish removal quality.</p>')
            for ident in case['held_out']:
                blocks.append('<details'+(' open' if ident==case['held_out'][0] else '')+'><summary>Held-out camera '+esc(ident)+'</summary><div class="pair">')
                for phase in ['before','after']:blocks.append('<figure>'+img(run/'rgb'/phase/(ident+'.png'),phase+' '+ident)+'<figcaption>'+phase.title()+' removal</figcaption></figure>')
                blocks.append('</div></details>')
        refs=a.evaluation/case['id']/'references'
        if refs.exists():
            blocks.append('<details><summary>Reference-mask review</summary><div class="triple">')
            for f in sorted(refs.glob('*_overlay.png')):blocks.append('<figure>'+img(f,'Pseudo-reference '+f.stem)+'<figcaption>'+esc(f.stem)+'</figcaption></figure>')
            blocks.append('</div></details>')
        deck_cases.append(case)
    blocks.append('<h2>Sources and interpretation</h2><p><a href="https://github.com/umangi-jain/gaussiancut">GaussianCut</a> performs the native edit. <a href="https://github.com/facebookresearch/segment-anything">SAM</a> segments views. <a href="https://github.com/IDEA-Research/Grounded-SAM-2">GroundedSAM2</a> probes residual object evidence. <a href="https://arxiv.org/html/2508.11431v3">Remove360, Appendix I</a> defines the evaluation setting.</p><p>The Python action harness and geometric mask gates are our implementation. 3DGSNav and AREA3D motivate the action/tool separation. Their code was not executed here. No RL, robot movement, or inpainting is included.</p><footer>Generated from saved experiment records. Every reported target and policy remains visible, including failures.</footer></main></html>')
    final=HERE/'V2 Results.html';final.write_text('\n'.join(blocks),encoding='utf-8')
    (HERE/'v2_presentation_data.json').write_text(json.dumps({'cases':deck_cases,'rows':rows,'run':str(a.run.resolve()),'evaluation':str(a.evaluation.resolve()),'report':report},indent=2))
    print(final)


if __name__=='__main__':main()
