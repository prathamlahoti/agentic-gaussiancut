"""Small invariants which catch pose/depth and wrong-object acceptance errors."""
import unittest
import numpy as np
from improved import backproject,projection,mask_quality


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rec={'width':100,'height':80,'fx':100,'fy':100,'cx':50,'cy':40,'world_to_camera':np.eye(4).tolist()}
        self.depth=np.full((80,100),2.,dtype=float)
        self.mask=np.zeros((80,100),bool);self.mask[20:60,30:70]=True
        self.points=backproject(self.rec,self.mask,self.depth)

    def test_round_trip(self):
        proj=projection(self.points,self.rec,self.depth)
        self.assertAlmostEqual(proj['inside'],1.)
        q=mask_quality(self.mask,proj,self.rec,self.depth,self.rec,self.mask,self.depth)
        self.assertTrue(q['accepted']);self.assertAlmostEqual(q['reverse_agreement'],1.)

    def test_floor_mask_rejected(self):
        wrong=np.zeros_like(self.mask);wrong[60:75,25:75]=True
        proj=projection(self.points,self.rec,self.depth)
        q=mask_quality(wrong,proj,self.rec,self.depth,self.rec,self.mask,self.depth)
        self.assertFalse(q['accepted'])

    def test_occlusion_not_counted_as_evidence(self):
        closer=np.ones_like(self.depth)
        self.assertIsNone(projection(self.points,self.rec,closer))

    def test_known_translation(self):
        moved=dict(self.rec);t=np.eye(4);t[0,3]=-.2;moved['world_to_camera']=t.tolist()
        a=projection(self.points,self.rec,self.depth)['pixels']
        b=projection(self.points,moved,self.depth)['pixels']
        np.testing.assert_allclose(a[:,0]-b[:,0],10)


if __name__=='__main__':unittest.main()
