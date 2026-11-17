import base64,json,tempfile,os
username=os.environ["KAGGLE_USERNAME"]
from pathlib import Path
here=Path(__file__).resolve().parent
files=[here/x for x in ['improved.py','benchmark_metrics.py']]+list((here/'vendor/remove360/remove360').rglob('*.py'))
payload={str(p.relative_to(here)).replace('\\','/'):base64.b64encode(p.read_bytes()).decode() for p in files}
boot=f'''import base64,sys
from pathlib import Path
source=Path('/kaggle/working/source');source.mkdir(exist_ok=True)
for name,data in {payload!r}.items():
    p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
sys.path.insert(0,str(source))
'''
script=boot+(here/'evaluate_gpu.py').read_text();compile(script,'run.py','exec')
out=Path(tempfile.gettempdir())/'agentic-gaussiancut-evaluation';out.mkdir(exist_ok=True)
(out/'run.py').write_text(script,encoding='utf-8')
(out/'kernel-metadata.json').write_text(json.dumps({'id':f'{username}/agentic-gaussiancut-evaluation','title':'Agentic GaussianCut evaluation','code_file':'run.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':True,'dataset_sources':[],'competition_sources':[],'kernel_sources':[f'{username}/agentic-gaussiancut-v2',f'{username}/agentic-gaussiancut-room-pilot']}))
print(out)
