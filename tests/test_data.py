import unittest

import numpy as np

from neurotrain.data import load_dataset, prepare_data


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_dataset()
        cls.prepared = prepare_data(cls.frame)

    def test_dataset_contract(self):
        self.assertEqual(self.frame.shape, (569, 31))
        self.assertEqual(set(self.frame["diagnosis"].unique()), {"B", "M"})
        self.assertFalse(self.frame.isna().any().any())

    def test_split_is_complete_and_disjoint(self):
        data = self.prepared
        total = len(data.y_train) + len(data.y_val) + len(data.y_test)
        self.assertEqual(total, len(self.frame))
        train_idx = set(data.X_train_raw.index)
        val_idx = set(data.X_val_raw.index)
        test_idx = set(data.X_test_raw.index)
        self.assertTrue(train_idx.isdisjoint(val_idx))
        self.assertTrue(train_idx.isdisjoint(test_idx))
        self.assertTrue(val_idx.isdisjoint(test_idx))

    def test_scaler_is_fitted_only_on_train(self):
        means = self.prepared.X_train.mean(axis=0)
        self.assertTrue(np.allclose(means, 0.0, atol=1e-5))


if __name__ == "__main__":
    unittest.main()

