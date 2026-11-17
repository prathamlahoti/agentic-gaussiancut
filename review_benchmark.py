"""Apply one documented pre-image reference review; retain raw GPU evaluation."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from improved import overlay
from benchmark_metrics import score_arm
HERE=Path(__file__).resolve().parent
raw=HERE/'evaluation_output/evaluation'
reviewed=HERE/'evaluation_reviewed/evaluation'
shutil.copytree(raw,reviewed,dirs_exist_ok=True)
refs=reviewed/'counter_plant/references'
path=refs/'v015.png'
original=np.asarray(Image.open(path))>0
segments=np.load(refs/'v015_sam.npz')['masks']
# In this saved automatic-mask set, segment 67 is the visible white pot front.
# Chosen by reviewing the ORIGINAL image, never the edited scene or scores.
addition=segments[67]
target=original|addition
Image.fromarray(target.astype('uint8')*255).save(path)
pre=Image.open(HERE/'v2_output/experiment/v2/scenes/counter/v015.png').convert('RGB')
overlay(pre,target).save(refs/'v015_overlay.png')
record={'case':'counter_plant','view':'v015','source':'pre-edit SAM automatic mask set','segment_index':67,'reason':'Target is plant including pot; initial pseudo-reference omitted visible white pot front','added_pixels':int((target&~original).sum()),'original_mask_sha256':hashlib.sha256((raw/'counter_plant/references/v015.png').read_bytes()).hexdigest(),'corrected_mask_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'applied_equally_to':['fixed','vlm'],'raw_evaluation_preserved':str(raw)}
(refs/'manual_review.json').write_text(json.dumps(record,indent=2))
summary=json.loads((raw/'summary.json').read_text())
for row in summary:
    if row['case']=='counter_plant':
        arm=reviewed/row['case']/row['policy']
        shutil.copy2(path,arm/'masks/v015.png')
        row['metrics']=score_arm(arm)
        row['metrics']['reference_review']=record
        (arm/'metrics.json').write_text(json.dumps(row['metrics'],indent=2))
(reviewed/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(record,indent=2))
