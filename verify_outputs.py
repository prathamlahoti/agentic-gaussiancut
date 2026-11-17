"""Verify downloaded render hashes and edit/evaluation camera separation."""
import hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent
root=here/'v2_output/experiment'
cfg=json.loads((root/'v2_config.json').read_text())
verified=[]
for case in cfg['cases']:
    for policy in ['fixed','vlm']:
        arm=root/'v2'/case['id']/policy
        manifest=root/'removal_pipeline/runs'/(case['id']+'_'+policy)/'manifest.json'
        if not manifest.exists():
            verified.append({'case':case['id'],'policy':policy,'status':'no edited output'})
            continue
        m=json.loads(manifest.read_text());selection=json.loads((arm/'selection.json').read_text())
        assert not set(selection['accepted']) & set(case['held_out'])
        assert 0<m['gaussians_after']<m['gaussians_before']
        for row in m['outputs']:
            p=manifest.parent/row['path']
            assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],str(p)
        verified.append({'case':case['id'],'policy':policy,'status':'verified','render_files':len(m['outputs']),'removed':m['gaussians_before']-m['gaussians_after']})
(here/'v2_integrity.json').write_text(json.dumps(verified,indent=2))
print(json.dumps(verified,indent=2))
