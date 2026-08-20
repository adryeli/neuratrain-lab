import unittest

import numpy as np

from neurotrain.evaluation import classification_metrics


class EvaluationTests(unittest.TestCase):
    def test_known_confusion_matrix(self):
        y_true = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.8, 0.9, 0.2])
        metrics = classification_metrics(y_true, probabilities, threshold=0.5)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["sensitivity"], 0.5)
        self.assertEqual(metrics["specificity"], 0.5)

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            classification_metrics(np.array([0, 1]), np.array([0.1, 0.9]), threshold=1)


if __name__ == "__main__":
    unittest.main()

