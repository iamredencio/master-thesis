"""Unit checks for model construction and trained research artifact."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import FEATURES, LABEL_END, load_data, normalize_name, rookie_age, run

class ModelChecks(unittest.TestCase):
    def test_name_normalization(self):
        self.assertEqual(normalize_name("José O'Neal Jr."), "joseonealjr")
    def test_age_cannot_use_future_career_outcome(self):
        self.assertEqual(rookie_age("September 2, 1997", 2016), 19)
    def test_only_pregame_features(self):
        self.assertTrue({"min","pts","gp","eff","age"}.issubset(set(FEATURES)))
        self.assertNotIn("year_end", FEATURES)
        self.assertNotIn("lastseason", FEATURES)
    def test_censored_player_data(self):
        X, y, years, catalog = load_data()
        self.assertEqual(len(X), 1280)
        self.assertLessEqual(int(years.max()), LABEL_END)
        self.assertTrue(all(p["observedFive"] is None for p in catalog if p["year"] > LABEL_END))
    def test_holdout_beats_baseline(self):
        model, _ = run()
        self.assertGreater(model["metrics"]["heldOut"]["auc"], .7)
        self.assertLess(model["metrics"]["heldOut"]["brier"], model["metrics"]["baselineHeldOut"]["brier"])

if __name__ == "__main__":
    unittest.main()
