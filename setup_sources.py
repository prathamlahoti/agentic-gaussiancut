import subprocess
from pathlib import Path
p=Path(__file__).resolve().parent/'vendor/remove360'
p.parent.mkdir(exist_ok=True)
if not p.exists(): subprocess.run(['git','clone','https://github.com/simonakocour/anything_left.git',str(p)],check=True)
subprocess.run(['git','-C',str(p),'checkout','dd0aad895b4d4dce75bd9f2462ad36114e813bd6'],check=True)
