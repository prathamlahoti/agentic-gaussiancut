"""Build a new private job; preserves v1 and its outputs."""
import base64,json,tempfile,os
username=os.environ["KAGGLE_USERNAME"]
from pathlib import Path
here=Path(__file__).resolve().parent
old=here
payload={f'removal_pipeline/{x}':base64.b64encode((old/'removal_pipeline'/x).read_bytes()).decode() for x in ['gpu_adapter.py','runner.py','upstream_lock.json']}
payload['v2_config.json']=base64.b64encode((here/'v2_config.json').read_bytes()).decode()
driver=(old/'driver.py').read_text()
driver=driver[:driver.index('def one_scene(')]
boot=f'''import base64
from pathlib import Path
for name,data in {payload!r}.items():
    path=Path('/kaggle/working/experiment')/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(base64.b64decode(data))
'''
script=boot+driver+(here/'improved.py').read_text()
compile(script,'run.py','exec')
out=Path(tempfile.gettempdir())/'agentic-gaussiancut-v2';out.mkdir(exist_ok=True)
(out/'run.py').write_text(script,encoding='utf-8')
(out/'kernel-metadata.json').write_text(json.dumps({'id':f'{username}/agentic-gaussiancut-v2','title':'Agentic GaussianCut v2','code_file':'run.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':True,'dataset_sources':[],'competition_sources':[],'kernel_sources':[]}))
print(out)
