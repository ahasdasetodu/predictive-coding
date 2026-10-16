import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

import main
from pc_core.byproduct import ConditionWeights
from pc_core.experiment import (
    TRAINERS,
    ByproductSpec,
    ClueBitsSpec,
    ExperimentSpec,
    Finding,
    code_identity,
    dump_spec,
    load_findings,
    load_spec,
    parse_cli,
    parse_findings,
    parse_spec,
    run_spec,
    spec_argv,
    train_byproduct,
    train_mnist,
    training_argv,
)
from pc_core.messages import Degradation
from pc_core.trainer import BatchReport, EpochReport

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "experiments" / "specs" / "balanced-256-128.yaml"
CLUE_BITS = ROOT / "experiments" / "specs" / "clue-bits.yaml"
HUB_CLEAN = ROOT / "experiments" / "specs" / "hub-clean.yaml"


class SpecTests(unittest.TestCase):
    """Verify that a spec loads, rejects unknown keys, and survives a round trip."""

    def test_unknown_key_is_rejected(self) -> None:
        """Refuse a key the spec format does not define."""
        text = EXAMPLE.read_text(encoding="utf-8") + "not_a_field: 1\n"
        with self.assertRaises(ValueError) as caught:
            parse_spec(text)
        self.assertIn("not_a_field", str(caught.exception))

    def test_round_trip(self) -> None:
        """Dumping a loaded spec and loading it again returns the same spec."""
        spec = load_spec(EXAMPLE)
        dumped = dump_spec(spec)
        self.assertIn("infer_lr:", dumped)
        self.assertIn("eval_lr:", dumped)
        self.assertIn("weight_lr:", dumped)
        self.assertIn("min_lr:", dumped)
        self.assertEqual(parse_spec(dumped), spec)
        overridden = replace(spec, infer_lr=0.05, eval_lr=0.02, weight_lr=0.004, min_lr=1e-4)
        self.assertEqual(parse_spec(dump_spec(overridden)), overridden)

    def test_infer_steps_is_required_and_defaults_to_one(self) -> None:
        """Require infer_steps in a spec, and keep the command-line default at 1."""
        fields = _example_fields()
        del fields["infer_steps"]
        with self.assertRaises(ValueError) as caught:
            parse_spec(yaml.safe_dump(fields))
        self.assertIn("infer_steps", str(caught.exception))
        self.assertEqual(parse_cli([]).infer_steps, 1)

    def test_argv_reproduces_spec(self) -> None:
        """The recorded command parses back to the same training spec."""
        spec = load_spec(EXAMPLE)
        argv = training_argv(spec)
        self.assertEqual(argv[:3], ["uv", "run", "main.py"])
        self.assertEqual(parse_cli(argv[3:]), replace(spec, name="cli"))

    def test_example_is_the_claimed_configuration(self) -> None:
        """The checked-in spec is the balanced 256-128 run, without an invented accuracy."""
        spec = load_spec(EXAMPLE)
        self.assertEqual(spec.regime, "balanced")
        self.assertEqual(spec.init, "forward")
        self.assertEqual(spec.hidden, (256, 128))
        self.assertEqual(spec.epochs, 8)
        self.assertEqual(spec.infer_steps, 1)
        self.assertEqual(spec.eval_steps, 1)
        self.assertEqual(spec.seed, 0)
        text = EXAMPLE.read_text(encoding="utf-8")
        self.assertIn("experiments/results/balanced-256-128-20261005T210648303658Z", text)
        self.assertNotIn("87.2", text)

    def test_main_uses_the_runner_training_path(self) -> None:
        """main.py and the spec runner call the same training function."""
        self.assertIs(main.train_mnist, train_mnist)
        self.assertIs(TRAINERS["mnist"], train_mnist)
        self.assertIsNone(run_spec.__kwdefaults__["train"])

    def test_clue_bits_spec_round_trips_with_defaults(self) -> None:
        """A clue-bits spec parses with default sampler and reference settings and survives a round trip."""
        spec = load_spec(CLUE_BITS)
        self.assertIsInstance(spec, ClueBitsSpec)
        self.assertEqual(spec.kind, "clue-bits")
        self.assertEqual(spec.sampler.positive, 0.5)
        self.assertEqual(spec.reference.hidden, 256)
        self.assertEqual(parse_spec(dump_spec(spec)), spec)
        self.assertEqual(spec_argv(spec)[-1], "experiments/specs/clue-bits.yaml")
        with self.assertRaises(ValueError):
            parse_spec("kind: clue-bits\nname: x\nseed: 0\nsampler: {bogus: 1}\n")
        with self.assertRaises(ValueError):
            parse_spec("kind: unknown\nname: x\nseed: 0\n")

    def test_byproduct_spec_round_trips_and_validates(self) -> None:
        """The hub spec fills defaults, survives a round trip, and rejects bad degradation."""
        spec = load_spec(HUB_CLEAN)
        self.assertIsInstance(spec, ByproductSpec)
        self.assertEqual((spec.z_dim, spec.h_dim, spec.epochs), (128, 256, 8))
        self.assertEqual((spec.topology, spec.h_msg, spec.h_lab), ("hub", 128, 128))
        self.assertTrue(spec.share_init)
        self.assertEqual(spec.degradation, Degradation("none", 0.0))
        self.assertEqual(spec.conditions, ConditionWeights(1.0, 1.0, 1.0))
        self.assertTrue(spec.balance)
        self.assertEqual(spec.eval_curve, (1, 5, 10, 30))
        self.assertEqual(parse_spec(dump_spec(spec)), spec)
        self.assertIs(TRAINERS["byproduct"], train_byproduct)
        base = "kind: byproduct\nname: x\nseed: 0\ngraph: {z: 4, h_img: 4}\nepochs: 1\ninfer_steps: 1\neval_steps: 1\n"
        self.assertEqual(parse_spec(base + "degrade: {kind: noise, level: 0.5}\n").degradation, Degradation("noise", 0.5))
        self.assertEqual(parse_spec(base + "degrade: {kind: occlude, level: 8}\n").degradation, Degradation("occlude", 8.0))
        for bad in ("degrade: {kind: noise}\n", "degrade: {kind: occlude, level: 40}\n", "degrade: {kind: blur, level: 1}\n"):
            with self.assertRaises(ValueError):
                parse_spec(base + bad)
        with self.assertRaises(ValueError):
            parse_spec(base + "degrade: {kind: none}\nconditions: {image: 0, msg: 0, both: 0}\n")
        with self.assertRaises(ValueError):
            parse_spec(
                "kind: byproduct\nname: x\nseed: 0\ngraph: {z: 4, h_img: 4, topology: star}\n"
                "epochs: 1\ninfer_steps: 1\neval_steps: 1\ndegrade: {kind: none}\n"
            )
        consensus = parse_spec(
            "kind: byproduct\nname: x\nseed: 0\n"
            "graph: {topology: consensus, z: 4, h_img: 8, h_msg: 6, h_lab: 5}\n"
            "epochs: 1\ninfer_steps: 10\neval_steps: 10\ndegrade: {kind: occlude, level: 14}\n"
        )
        self.assertEqual(consensus.topology, "consensus")
        self.assertTrue(consensus.share_init)
        self.assertEqual((consensus.h_msg, consensus.h_lab, consensus.infer_steps), (6, 5, 10))
        self.assertEqual(parse_spec(dump_spec(consensus)), consensus)
        own_init = parse_spec(
            "kind: byproduct\nname: x\nseed: 0\n"
            "graph: {topology: consensus, z: 4, h_img: 8, share_init: false}\n"
            "epochs: 1\ninfer_steps: 10\neval_steps: 10\ndegrade: {kind: occlude, level: 14}\n"
        )
        self.assertFalse(own_init.share_init)
        self.assertEqual(parse_spec(dump_spec(own_init)), own_init)

    def test_train_mnist_settles_one_synthetic_batch(self) -> None:
        """The shared path runs Trainer on one batch and returns the regime's metrics."""
        spec = replace(load_spec(EXAMPLE), name="smoke", hidden=(4,), epochs=1)
        images = torch.randn(2, 28 * 28)
        labels = torch.tensor([0, 1])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)

        def fake_dataset(batch_size: int = 256) -> tuple[DataLoader, DataLoader]:
            """Substitute a two-image loader for MNIST."""
            return loader, loader

        with patch("pc_core.experiment.mnist.load_dataset", fake_dataset):
            reports = train_mnist(spec)

        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].batches[0].batch_idx, 0)
        self.assertIn("init_energy", reports[0].batches[0].metrics)
        self.assertIn("final_energy", reports[0].batches[0].metrics)
        self.assertGreaterEqual(reports[0].validation_accuracy, 0.0)
        self.assertLessEqual(reports[0].validation_accuracy, 100.0)


class ResultTests(unittest.TestCase):
    """Verify that a run writes provenance and the metrics the trainer returned."""

    def test_result_records_provenance_and_trainer_metrics(self) -> None:
        """Write the resolved spec, logged metrics, and provenance, completed only at the end."""
        spec = load_spec(EXAMPLE)
        report = EpochReport(
            epoch=0,
            train_accuracy=10.0,
            validation_accuracy=12.5,
            batches=(BatchReport(0, 1.5, {"init_energy": 0.4, "final_energy": 0.2}),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            seen_running = False

            def fake(received: ExperimentSpec) -> list[EpochReport]:
                """Return a fixed report after checking the run is not yet completed."""
                nonlocal seen_running
                self.assertEqual(received, spec)
                matches = list(results.glob(f"{spec.name}-*/result.yaml"))
                self.assertEqual(len(matches), 1)
                running = yaml.safe_load(matches[0].read_text(encoding="utf-8"))
                self.assertEqual(running["status"], "running")
                self.assertIsNone(running["metrics"])
                seen_running = True
                return [report]

            directory = run_spec(spec, train=fake, results_root=results, project_root=ROOT)

            self.assertTrue(seen_running)
            saved = load_spec(directory / "spec.yaml")
            self.assertEqual(saved, spec)
            record = yaml.safe_load((directory / "result.yaml").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "completed")
            epoch = record["metrics"]["epochs"][0]
            self.assertEqual(epoch["train_accuracy"], 10.0)
            self.assertEqual(epoch["validation_accuracy"], 12.5)
            self.assertEqual(epoch["batches"][0]["loss"], 1.5)
            self.assertEqual(epoch["batches"][0]["metrics"], {"init_energy": 0.4, "final_energy": 0.2})
            _assert_provenance(self, record["provenance"], spec)

    def test_failed_run_is_not_completed(self) -> None:
        """Leave a run unfinished when training raises."""
        spec = load_spec(EXAMPLE)

        def fake(received: ExperimentSpec) -> list[EpochReport]:
            """Stop before a result can be marked completed."""
            raise RuntimeError("stopped")

        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            with self.assertRaises(RuntimeError):
                run_spec(spec, train=fake, results_root=results, project_root=ROOT)
            matches = list(results.glob(f"{spec.name}-*/result.yaml"))
            self.assertEqual(len(matches), 1)
            record = yaml.safe_load(matches[0].read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "failed")

    def test_code_identity_hashes_sources_outside_a_git_repo(self) -> None:
        """Say so, and hash pc_core plus main.py, when the directory is not a git repo."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "proj"
            (root / "pc_core").mkdir(parents=True)
            (root / "pc_core" / "graph.py").write_text("x = 1\n", encoding="utf-8")
            (root / "main.py").write_text("y = 2\n", encoding="utf-8")
            identity = code_identity(root)
        self.assertEqual(identity["vcs"], "none")
        self.assertEqual(identity["detail"], "not a git repository")
        self.assertTrue(identity["source_hash"].startswith("sha256:"))


class FindingsTests(unittest.TestCase):
    """Verify the findings index has the required fields and ties accuracies to runs."""

    def test_entries_have_required_fields(self) -> None:
        """Every entry has an id, claim, status, evidence, and note."""
        findings = load_findings()
        self.assertGreater(len(findings), 0)
        self.assertEqual(len({item.id for item in findings}), len(findings))
        for item in findings:
            self.assertTrue(item.id.strip())
            self.assertTrue(item.claim.strip())
            self.assertIn(item.status, ("confirmed", "contradicted", "unverified", "historical"))
            self.assertTrue(item.evidence.strip())
            self.assertTrue(item.note.strip())

    def test_missing_field_is_rejected(self) -> None:
        """Refuse a finding that omits a required field."""
        text = """
findings:
  - id: x
    claim: y
    status: historical
    evidence: no artifact
"""
        with self.assertRaises(ValueError) as caught:
            parse_findings(text)
        self.assertIn("note", str(caught.exception))

    def test_accuracy_is_confirmed_only_by_a_completed_run(self) -> None:
        """A confirmed accuracy matches the best validation accuracy in its result directory."""
        findings = load_findings()
        accuracies = [item for item in findings if _is_accuracy(item)]
        self.assertTrue(accuracies)
        for item in accuracies:
            if item.status == "unverified":
                self.assertEqual(item.evidence, "no artifact")
                continue
            self.assertEqual(item.status, "confirmed")
            claimed = _claimed_percent(item.claim)
            candidates = _reported_percentages(ROOT / item.evidence)
            self.assertTrue(
                any(abs(claimed - value) <= 0.05 for value in candidates),
                f"{item.id}: {claimed} is not reported in {item.evidence}",
            )
        for item in findings:
            if item.status != "confirmed":
                continue
            self.assertNotEqual(item.evidence, "no artifact")


def _example_fields() -> dict[str, object]:
    """The example spec as a mapping, comments dropped."""
    loaded = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise AssertionError("example spec is not a mapping")
    return {str(key): value for key, value in loaded.items()}


def _assert_provenance(case: unittest.TestCase, provenance: object, spec: ExperimentSpec) -> None:
    """Require seed, timestamp, argv, and either a git commit or a source hash."""
    if not isinstance(provenance, dict):
        case.fail("provenance is not a mapping")
    case.assertEqual(provenance["seed"], spec.seed)
    case.assertIsInstance(provenance["timestamp"], str)
    case.assertTrue(provenance["timestamp"])
    case.assertEqual(provenance["argv"], training_argv(spec))
    code = provenance["code"]
    if not isinstance(code, dict):
        case.fail("code identity is not a mapping")
    if code.get("vcs") == "git":
        case.assertIsInstance(code.get("commit"), str)
        case.assertEqual(len(code["commit"]), 40)
        return
    case.assertEqual(code.get("detail"), "not a git repository")
    case.assertIsInstance(code.get("source_hash"), str)
    case.assertTrue(str(code["source_hash"]).startswith("sha256:"))


def _is_accuracy(finding: Finding) -> bool:
    """Whether a claim is stating a classification accuracy."""
    claim = finding.claim.lower()
    return "accuracy" in claim or "accuracies" in claim or "%" in finding.claim


def _claimed_percent(claim: str) -> float:
    """The validation accuracy stated in a finding claim."""
    match = re.search(r"(\d+\.\d+)%", claim)
    if match is None:
        raise AssertionError(f"no percentage in claim: {claim}")
    return float(match.group(1))


def _reported_percentages(result_dir: Path) -> list[float]:
    """Percentages a completed result directory reports: the best validation accuracy, plus every
    number in its evaluation or clue table."""
    document = yaml.safe_load((result_dir / "result.yaml").read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise AssertionError(f"{result_dir} result is not a mapping")
    if document.get("status") != "completed":
        raise AssertionError(f"{result_dir} is not completed")
    metrics = document.get("metrics")
    if not isinstance(metrics, dict):
        raise AssertionError(f"{result_dir} has no metrics")
    epochs = metrics.get("epochs")
    if not isinstance(epochs, list):
        raise AssertionError(f"{result_dir} has no epochs")
    values = [max(float(epoch["validation_accuracy"]) for epoch in epochs)] if epochs else []
    for artifact in ("evaluation.yaml", "table.yaml"):
        path = result_dir / artifact
        if path.is_file():
            values.extend(_numbers(yaml.safe_load(path.read_text(encoding="utf-8"))))
    if not values:
        raise AssertionError(f"{result_dir} reports no percentages")
    return values


def _numbers(node: object) -> list[float]:
    """Every number nested anywhere in a parsed YAML document."""
    if isinstance(node, bool):
        return []
    if isinstance(node, (int, float)):
        return [float(node)]
    if isinstance(node, dict):
        return [number for value in node.values() for number in _numbers(value)]
    if isinstance(node, list):
        return [number for value in node for number in _numbers(value)]
    return []


if __name__ == "__main__":
    unittest.main()
