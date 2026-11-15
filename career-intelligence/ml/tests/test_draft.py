"""Historical conditional draft-path invariants and chronological holdouts."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from draft_path import make,load

class DraftPathQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=make()
    def test_excludes_unmatured_cohorts(self):
        records,_,_=load()
        self.assertEqual(len(records),1160)
        self.assertLessEqual(max(r["year"] for r in records),2012)
    def test_holdout_is_temporal(self):
        self.assertEqual(self.model["horizons"]["five"]["n"]["test"],201)
        self.assertEqual(self.model["horizons"]["ten"]["n"]["test"],199)
    def test_beats_prevalence_only(self):
        for key in ("five","ten"):
            m=self.model["horizons"][key]["test"]
            self.assertLess(m["brier"],m["baseline"])
    def test_selection_and_horizon_consistency(self):
        five=self.model["horizons"]["five"]["bands"]
        ten=self.model["horizons"]["ten"]["bands"]
        self.assertGreater(five[0]["prob"],five[-1]["prob"])
        for f,t in zip(five,ten):
            self.assertLessEqual(t["prob"],f["prob"])

if __name__=="__main__":
    unittest.main()
