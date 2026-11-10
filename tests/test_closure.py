"""Closure adapter contract tests; fixtures are synthetic, not trained models."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from standalone.closure import ClosureClassifier
from standalone.models import Detection


class ClosureTests(unittest.TestCase):
    def fixture(self, root, values, **config_updates):
        node = helper.make_node("Constant", [], ["logits"],
                                value=numpy_helper.from_array(np.array([values], np.float32)))
        graph = helper.make_graph([node], "synthetic_test_only", [
            helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])], [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
        onnx.save(model, str(root / "fixture.onnx"))
        config = dict(model="fixture.onnx", classes=["open", "closed", "unclear"],
                      confidence=.85, mean=[.485, .456, .406], std=[.229, .224, .225], output="logits")
        config.update(config_updates)
        path = root / "closure.json"
        path.write_text(json.dumps(config))
        classifier = ClosureClassifier(path)
        frame = np.full((100, 100, 3), 180, np.uint8)
        bag = Detection(np.full((100, 100), 255, np.uint8), (10, 10, 90, 90), .95)
        return classifier, frame, bag

    def test_missing_model_configuration_never_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            classifier = ClosureClassifier(Path(tmp) / "absent.json")
            self.assertFalse(classifier.ready)
            self.assertEqual(classifier.classify(None, None).status, "model_missing")

    def test_configured_model_and_class_order_are_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            classifier, frame, bag = self.fixture(Path(tmp), [8., 0., 0.], classes=["closed", "unclear", "open"])
            self.assertTrue(classifier.ready, classifier.problem)
            result = classifier.classify(frame, bag)
            self.assertEqual(result.status, "closed")
            self.assertGreater(result.confidence, .99)

    def test_open_and_low_confidence_do_not_pass_as_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            for values, expected in (([8., 0., 0.], "open"), ([0., 0., 0.], "unclear")):
                classifier, frame, bag = self.fixture(Path(tmp), values)
                self.assertEqual(classifier.classify(frame, bag).status, expected)

    def test_invalid_class_mapping_is_a_configuration_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            classifier, _, _ = self.fixture(Path(tmp), [0., 8., 0.], classes=["open", "closed", "closed"])
            self.assertFalse(classifier.ready)
            self.assertEqual(classifier.classify(None, None).status, "model_error")

    def test_missing_weights_do_not_fall_back_to_strip_only_sealing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            classifier, frame, bag = self.fixture(root, [0., 8., 0.])
            (root / "fixture.onnx").unlink()
            classifier = ClosureClassifier(root / "closure.json")
            self.assertEqual(classifier.classify(frame, bag).status, "model_missing")

    def test_nonfinite_output_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            classifier, frame, bag = self.fixture(Path(tmp), [0., float("nan"), 0.])
            self.assertEqual(classifier.classify(frame, bag).status, "model_error")

    def test_probability_output_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            classifier, frame, bag = self.fixture(Path(tmp), [.02, .96, .02], output="probabilities")
            self.assertEqual(classifier.classify(frame, bag).status, "closed")

    def test_no_packet_cannot_be_classified_as_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            classifier, frame, _ = self.fixture(Path(tmp), [0., 8., 0.])
            self.assertEqual(classifier.classify(frame, None).status, "unclear")


if __name__ == "__main__":
    unittest.main()
