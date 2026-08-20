import unittest

from neurotrain.config import TrainingConfig
from neurotrain.modeling import build_dense_classifier, tensorflow_available


class ModelingTests(unittest.TestCase):
    def test_invalid_dropout_is_rejected(self):
        with self.assertRaises(ValueError):
            TrainingConfig(dropout_rate=1.0).validate()

    @unittest.skipUnless(tensorflow_available(), "TensorFlow is not installed")
    def test_model_output_shape(self):
        model = build_dense_classifier(30, TrainingConfig())
        self.assertEqual(model.output_shape, (None, 1))


if __name__ == "__main__":
    unittest.main()

