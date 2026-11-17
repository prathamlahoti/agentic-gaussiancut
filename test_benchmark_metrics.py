import unittest,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from benchmark_metrics import structural_similarity,depth_pair,evaluate_depth_scene


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.m=np.zeros((64,64),bool);self.m[20:44,20:44]=True

    def test_structural_empty_sets_are_retained(self):
        self.assertEqual(structural_similarity([],[],self.m)['sim_sam'],1)
        self.assertEqual(structural_similarity([self.m],[],self.m)['sim_sam'],0)
        self.assertEqual(structural_similarity([self.m],[self.m],self.m)['sim_sam'],1)

    def test_unmatched_segments_penalized(self):
        half=self.m.copy();half[:32]=False
        self.assertAlmostEqual(structural_similarity([self.m,half],[self.m],self.m)['sim_sam'],.5)

    def test_identical_and_changed_depth(self):
        for changed in [False,True]:
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);(p/'masks').mkdir()
                Image.fromarray(self.m.astype('uint8')*255).save(p/'masks/v.png')
                a=np.full((64,64),2.);b=a.copy()
                if changed:b[self.m]=4.
                depth_pair(a,b,p,'v')
                score=evaluate_depth_scene(tmp,verbose=False)[0]['acc_delta_depth']
                self.assertAlmostEqual(score,float(changed))


if __name__=='__main__':unittest.main()
