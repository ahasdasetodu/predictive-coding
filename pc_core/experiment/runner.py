import hashlib
import subprocess
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pc_core.experiment.spec import TRAINERS, Spec, dump_spec, spec_argv
from pc_core.experiment.yamlio import write_text, write_yaml
from pc_core.trainer import BatchReport, EpochReport

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "experiments" / "results"
RunStatus = Literal["running", "completed", "failed"]


def run_spec(
    spec: Spec,
    *,
    train: Callable[[Spec], Sequence[EpochReport]] | None = None,
    results_root: Path | None = None,
    project_root: Path | None = None,
) -> Path:
    """Run a spec and write its result directory. Status is completed only after training returns.

    Without `train`, the trainer registered for the spec's kind runs and may write
    artifacts into the result directory.
    """
    root = PROJECT_ROOT if project_root is None else project_root
    destination = RESULTS_ROOT if results_root is None else results_root
    started = datetime.now(timezone.utc)
    directory = destination / f"{spec.name}-{started.strftime('%Y%m%dT%H%M%S%fZ')}"
    directory.mkdir(parents=True)
    write_text(directory / "spec.yaml", dump_spec(spec))
    provenance = _provenance(spec, started, root)
    result_path = directory / "result.yaml"
    _write_result(result_path, "running", None, provenance, error=None)
    try:
        if train is None:
            reports = list(TRAINERS[spec.kind](spec, directory))
        else:
            reports = list(train(spec))
    except Exception as exc:
        _write_result(result_path, "failed", None, provenance, error=str(exc))
        raise
    _write_result(result_path, "completed", reports, provenance, error=None)
    return directory


def code_identity(root: Path) -> dict[str, str]:
    """Git commit when `root` is a repo; otherwise a hash of the training sources."""
    commit = _git_commit(root)
    if commit is None:
        return {
            "vcs": "none",
            "detail": "not a git repository",
            "source_hash": _source_hash(root),
        }
    identity = {"vcs": "git", "commit": commit, "dirty": str(_git_dirty(root)).lower()}
    # A dirty tree is not the commit. Hash the sources the run actually imported.
    if identity["dirty"] == "true":
        identity["source_hash"] = _source_hash(root)
    return identity


def _provenance(spec: Spec, started: datetime, root: Path) -> dict[str, object]:
    """Seed, time, command line, and code identity for one run."""
    return {
        "seed": spec.seed,
        "timestamp": started.isoformat(),
        "argv": spec_argv(spec),
        "code": code_identity(root),
    }


def _write_result(
    path: Path,
    status: RunStatus,
    reports: Sequence[EpochReport] | None,
    provenance: dict[str, object],
    error: str | None,
) -> None:
    """Write status, logged metrics, and provenance. Metrics appear only when completed."""
    document: dict[str, object] = {"status": status, "metrics": _metrics(reports), "provenance": provenance}
    if error is not None:
        document["error"] = error
    write_yaml(path, document)


def _metrics(reports: Sequence[EpochReport] | None) -> dict[str, object] | None:
    """Accuracies and the batch measurements Trainer logged, or nothing before the run finishes."""
    if reports is None:
        return None
    return {
        "epochs": [
            {
                "epoch": report.epoch,
                "train_accuracy": report.train_accuracy,
                "validation_accuracy": report.validation_accuracy,
                "batches": [_batch_document(batch) for batch in report.batches],
            }
            for report in reports
        ]
    }


def _batch_document(batch: BatchReport) -> dict[str, object]:
    """One logged batch: the index, loss, and regime metrics Trainer printed."""
    return {"batch_idx": batch.batch_idx, "loss": batch.loss, "metrics": dict(batch.metrics)}


def _git_commit(root: Path) -> str | None:
    """HEAD when `root` is its own git work tree."""
    if _git(root, ["rev-parse", "--is-inside-work-tree"]) != "true":
        return None
    toplevel = _git(root, ["rev-parse", "--show-toplevel"])
    if toplevel is None or Path(toplevel).resolve() != root.resolve():
        return None
    return _git(root, ["rev-parse", "HEAD"])


def _git_dirty(root: Path) -> bool:
    """Whether the work tree has staged, unstaged, or untracked changes."""
    status = _git(root, ["status", "--porcelain"])
    if status is None:
        return True
    return bool(status)


def _git(root: Path, args: list[str]) -> str | None:
    """Run one git query. Returns stripped stdout, or None when git fails."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _source_hash(root: Path) -> str:
    """SHA-256 of pc_core Python sources and main.py, in path order."""
    digest = hashlib.sha256()
    paths = sorted((root / "pc_core").rglob("*.py"))
    paths.append(root / "main.py")
    for path in paths:
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"
