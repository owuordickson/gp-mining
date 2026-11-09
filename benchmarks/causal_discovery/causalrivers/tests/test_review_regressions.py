"""Hermetic regressions for PR #232; never deserialize repository pickles."""

import hashlib
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd

from tools import integrity
from tools.scoring_tools import max_accuracy, score
from tools.tools import benchmarking, graph_to_label_tensor, preprocess_data, remove_trailing_nans


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.path = self.root / "datasets" / "sample.p"
        self.path.parent.mkdir()
        self.path.write_bytes(b"not a pickle: the unpickler is mocked")
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.manifest = self.root / "run_manifest.json"
        self.write_manifest({"filepath": "datasets/sample.p", "sha256": self.digest})
        for name, value in (("BENCHMARK_ROOT", self.root), ("MANIFEST_PATH", self.manifest)):
            mock = patch.object(integrity, name, value)
            mock.start()
            self.addCleanup(mock.stop)

    def write_manifest(self, entry):
        self.manifest.write_text(json.dumps({"data_runs": [entry]}), encoding="utf-8")

    def test_equivalent_paths_from_other_working_directory(self):
        with patch("tools.integrity.pickle.load", return_value="verified") as spy:
            for path in (
                Path("datasets/sample.p"),
                self.path,
                Path("datasets/../datasets/sample.p"),
            ):
                self.assertEqual(integrity.secure_load_pickle(path), "verified")
            self.assertEqual(spy.call_count, 3)

    def test_absolute_manifest_key(self):
        self.write_manifest({"filepath": str(self.path), "sha256": self.digest.upper()})
        with patch("tools.integrity.pickle.load", return_value="verified") as spy:
            integrity.secure_load_pickle("datasets/sample.p")
            spy.assert_called_once()

    def test_manifest_failures_never_unpickle(self):
        for content in (
            None,
            "{bad",
            "{}",
            '{"data_runs": []}',
            json.dumps({"data_runs": [{"filepath": "datasets/sample.p"}]}),
            json.dumps({"data_runs": [{"filepath": "datasets/sample.p", "sha256": "x"}]}),
            json.dumps({"data_runs": [{"filepath": "datasets/sample.p", "sha256": 123}]}),
            json.dumps({"data_runs": [{"filepath": "datasets/sample.p", "sha256": "0" * 64}]}),
        ):
            with self.subTest(content=content):
                if content is None:
                    self.manifest.unlink(missing_ok=True)
                else:
                    self.manifest.write_text(content, encoding="utf-8")
                with patch("tools.integrity.pickle.load") as spy:
                    with self.assertRaises(ValueError):
                        integrity.secure_load_pickle(self.path)
                    spy.assert_not_called()

    def test_missing_artifact_never_unpickles(self):
        self.path.unlink()
        with patch("tools.integrity.pickle.load") as spy:
            with self.assertRaises(FileNotFoundError):
                integrity.secure_load_pickle(self.path)
            spy.assert_not_called()

    def test_filename_does_not_alias_nested_artifact(self):
        self.write_manifest({"filename": "sample.p", "sha256": self.digest})
        with patch("tools.integrity.pickle.load") as spy:
            with self.assertRaises(ValueError):
                integrity.secure_load_pickle(self.path)
            spy.assert_not_called()

    def test_duplicate_and_outside_paths_refuse(self):
        entry = {"filepath": "datasets/sample.p", "sha256": self.digest}
        self.manifest.write_text(json.dumps({"data_runs": [entry, entry]}), encoding="utf-8")
        with patch("tools.integrity.pickle.load") as spy:
            for path in (self.path, self.root.parent / "outside.p"):
                with self.assertRaises(ValueError):
                    integrity.secure_load_pickle(path)
            spy.assert_not_called()

    def test_verified_bytes_are_the_bytes_loaded(self):
        def mutate_after_check(stream):
            self.path.write_bytes(b"changed on disk after verification")
            return stream.read()

        with patch("tools.integrity.pickle.load", side_effect=mutate_after_check):
            self.assertEqual(
                integrity.secure_load_pickle(self.path), b"not a pickle: the unpickler is mocked"
            )

    def test_rescoring_tamper_refuses_before_unpickling(self):
        from rescore import rescore

        run = self.root / "results/run"
        run.mkdir(parents=True)
        entries = {}
        for name in ("config.yaml", "preds.p", "runtime.csv", "scoring.csv"):
            path = run / name
            path.write_bytes(b"fixture, not a repository artifact")
            entries[name] = {
                "filepath": path.relative_to(self.root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        record = {
            "inputs": {"labels": {"filepath": "datasets/sample.p", "sha256": self.digest}},
            "artifacts": entries,
        }
        (run / "provenance.json").write_text(json.dumps(record))
        for name in entries:
            with self.subTest(name=name):
                target = run / name
                original = target.read_bytes()
                target.write_bytes(b"tampered")
                with patch("tools.integrity.pickle.load") as spy:
                    with self.assertRaisesRegex(ValueError, "Integrity check failed"):
                        rescore(run, self.root / "results/rescore")
                    spy.assert_not_called()
                target.write_bytes(original)

    def test_reconstructed_rescore_checks_node_order_and_scores_real_fixture(self):
        from rescore import rescore

        run = self.root / "results/reconstructed-run"
        run.mkdir(parents=True)
        (run / "config.yaml").write_text(
            "label_path: datasets/sample.p\nrestrict_to: -1\nremove_diagonal: true\nmethod:\n  name: fixture\n"
        )
        (run / "runtime.csv").write_text("runtime\n1\n")
        (run / "scoring.csv").write_text("Metric,fixture\nAUROC,0.5\n")
        graph = nx.DiGraph([(1, 2)])
        recovered = self.root / "datasets/recovered.p"
        recovered.write_bytes(pickle.dumps([graph]))
        self.write_manifest(
            {
                "filepath": "datasets/recovered.p",
                "sha256": hashlib.sha256(recovered.read_bytes()).hexdigest(),
            }
        )
        (self.root / "tools").mkdir()
        (self.root / "tools/scoring_tools.py").write_text("fixture scorer identity")
        # Recovery must work on a fresh checkout without the old ignored labels.
        self.path.unlink()
        for names in (["2", "1"], ["1", "2"]):
            predictions = [pd.DataFrame([[0, 0.2], [0.8, 0]], columns=names, index=names)]
            (run / "preds.p").write_bytes(pickle.dumps(predictions))
            artifacts = {
                name: {
                    "filepath": (run / name).relative_to(self.root).as_posix(),
                    "sha256": hashlib.sha256((run / name).read_bytes()).hexdigest(),
                }
                for name in ("config.yaml", "runtime.csv", "scoring.csv", "preds.p")
            }
            record = {
                "artifacts": artifacts,
                "inputs": {"labels": {"filepath": "datasets/sample.p", "sha256": self.digest}},
            }
            (run / "provenance.json").write_text(json.dumps(record))
            output = self.root / "results/fixture-rescore"
            if names == ["2", "1"]:
                with self.assertRaisesRegex(ValueError, "node identities and order"):
                    rescore(run, output, recovered)
                self.assertFalse(output.exists())
            else:
                with (
                    patch("rescore.environment_record", return_value={"mode": "test"}),
                    patch("rescore.source_record", return_value={"commit": "fixture"}),
                ):
                    scored = rescore(run, output, recovered)
                self.assertEqual(scored.loc["AUROC"].iloc[0], 1)


class PreprocessingTests(unittest.TestCase):
    def data(self, values):
        return pd.DataFrame(values, index=pd.date_range("2000-01-01", periods=3, freq="6h"))

    def test_constant_missing_mask_and_input_preserved(self):
        data = self.data({"constant": [5, np.nan, 5], "empty": [np.nan] * 3})
        before = data.copy(deep=True)
        out = preprocess_data(data, resolution="6h", normalize=True, interpolate=False)
        np.testing.assert_allclose(out["constant"], [0, np.nan, 0], equal_nan=True)
        self.assertTrue(out["empty"].isna().all())
        pd.testing.assert_frame_equal(data, before)

    def test_nonconstant_interpolation(self):
        data = self.data({"value": [2, np.nan, 6]})
        out = preprocess_data(data, resolution="6h", normalize=True, interpolate=True)
        np.testing.assert_allclose(out["value"], [0, 0.5, 1])

    def test_empty_and_no_fully_valid_rows_rejected(self):
        for data in (
            pd.DataFrame(),
            pd.DataFrame({"a": [1, np.nan], "b": [np.nan, 2]}),
            pd.DataFrame({"a": [np.inf, np.nan]}),
        ):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "Unusable sample"):
                remove_trailing_nans(data)

    def test_edge_trimming_preserves_interior_gap_and_input(self):
        data = pd.DataFrame({"a": [np.nan, 1, np.nan, 2, np.nan]})
        before = data.copy(deep=True)
        pd.testing.assert_frame_equal(remove_trailing_nans(data), data.iloc[1:4])
        pd.testing.assert_frame_equal(data, before)

    def test_unusable_selection_rejected_before_any_method(self):
        cfg = SimpleNamespace(method=SimpleNamespace(max_lag=1))
        good = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
        for samples in (
            [],
            [good, pd.DataFrame()],
            [good, pd.DataFrame({"a": [1, np.nan, 2]})],
            [good, pd.DataFrame({"a": [1, np.inf, 2]})],
            [good.iloc[:1]],
        ):
            with (
                self.subTest(samples=samples),
                patch("builtins.print"),
                patch("tools.baseline_methods.var_baseline") as method,
            ):
                with self.assertRaises(ValueError):
                    benchmarking(samples, cfg, method)
                method.assert_not_called()


class MethodAndScorerTests(unittest.TestCase):
    def test_accuracy_threshold_grid_preserves_first_best_threshold(self):
        threshold, accuracy = max_accuracy(np.array([0, 0, 1, 1]), np.array([0, 0.2, 0.8, 1]))
        self.assertAlmostEqual(threshold, 0.2)
        self.assertEqual(accuracy, 1)
        threshold, accuracy = max_accuracy(np.array([0, 1, 0]), np.zeros(3))
        self.assertEqual(threshold, 0)
        self.assertEqual(accuracy, 2 / 3)

    def test_tgraank_isolates_upstream_dataframe_mutation(self):
        from tools.baseline_methods import var_tgraank

        data = pd.DataFrame({"1": [1.0, 2.0, 3.0], "2": [3.0, 2.0, 1.0]})
        before = data.copy(deep=True)

        class MutatingTGRAANK:
            def __init__(self, frame, device):
                frame.loc[-1] = frame.columns

            def get_lagged_dependencies(self, max_lag):
                return np.zeros((2, 2))

        modules = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
            "so4gp.algorithms": SimpleNamespace(TGRAANK=MutatingTGRAANK),
        }
        with patch.dict(sys.modules, modules):
            prediction = var_tgraank(data, SimpleNamespace(max_lag=1))
        self.assertEqual(prediction.shape, (2, 2))
        pd.testing.assert_frame_equal(data, before)

    def test_var_preserves_input_and_produces_effect_by_cause_matrix(self):
        from tools.baseline_methods import var_baseline

        rng = np.random.default_rng(232)
        data = pd.DataFrame(
            rng.normal(size=(40, 2)),
            columns=["a", "b"],
            index=pd.date_range("2000", periods=40, freq="6h"),
        )
        before = data.copy(deep=True)
        config = SimpleNamespace(max_lag=1, var_absolute_values=True, map_to_summary_graph="max")
        prediction = var_baseline(data, config)
        self.assertEqual(prediction.shape, (2, 2))
        self.assertTrue(np.isfinite(prediction).all())
        pd.testing.assert_frame_equal(data, before)

    def test_orientation_metrics_and_label_immutability(self):
        graph = nx.DiGraph([("a", "b")])
        labels = graph_to_label_tensor(graph)
        np.testing.assert_array_equal(labels, [[0, 0], [1, 0]])
        prediction = np.array([[0, 0.2], [0.8, 0]])
        before_labels, before_prediction = labels.copy(), prediction.copy()
        result = score(prediction, labels)
        self.assertEqual(result.loc["AUROC"].iloc[0], 1)
        self.assertEqual(result.loc["Individual AUROC"].iloc[0], 1)
        np.testing.assert_array_equal(labels, before_labels)
        np.testing.assert_array_equal(prediction, before_prediction)

    def test_scoring_rejects_empty_mismatched_and_nonfinite_results(self):
        for predictions, labels in (
            (np.zeros((0, 2, 2)), np.zeros((0, 2, 2))),
            (np.zeros((2, 2)), np.zeros((3, 3))),
            (np.full((2, 2), np.nan), np.zeros((2, 2))),
        ):
            with self.subTest(shape=predictions.shape), self.assertRaises(ValueError):
                score(predictions, labels)


if __name__ == "__main__":
    unittest.main()
