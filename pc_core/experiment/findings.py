from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from pc_core.experiment.yamlio import as_text, reject_unknown, require, string_key_mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINDINGS_PATH = PROJECT_ROOT / "experiments" / "findings.yaml"
FindingStatus = Literal["confirmed", "contradicted", "unverified", "historical"]
_FINDING_FIELDS = {"id", "claim", "status", "evidence", "note"}


@dataclass(frozen=True)
class Finding:
    """One indexed claim and the artifact that supports it."""

    id: str
    claim: str
    status: FindingStatus
    evidence: str
    note: str


def load_findings(path: Path | None = None) -> tuple[Finding, ...]:
    """Load the findings index and reject an entry that is missing a field."""
    source = FINDINGS_PATH if path is None else path
    return parse_findings(source.read_text(encoding="utf-8"))


def parse_findings(text: str) -> tuple[Finding, ...]:
    """Parse a findings document and reject unknown keys or a bad status."""
    raw = string_key_mapping(yaml.safe_load(text), "findings file")
    reject_unknown(raw, {"findings"}, "findings file")
    require(raw, {"findings"}, "findings file")
    entries = raw["findings"]
    if not isinstance(entries, list):
        raise ValueError("findings must be a list")
    findings: list[Finding] = []
    seen: set[str] = set()
    for entry in entries:
        finding = _finding(entry)
        if finding.id in seen:
            raise ValueError(f"duplicate finding id {finding.id}")
        seen.add(finding.id)
        findings.append(finding)
    return tuple(findings)


def _finding(entry: object) -> Finding:
    """Parse one findings entry."""
    raw = string_key_mapping(entry, "finding")
    reject_unknown(raw, _FINDING_FIELDS, "finding")
    require(raw, _FINDING_FIELDS, "finding")
    return Finding(
        id=as_text(raw["id"], "id"),
        claim=as_text(raw["claim"], "claim"),
        status=_status(raw["status"]),
        evidence=as_text(raw["evidence"], "evidence"),
        note=as_text(raw["note"], "note"),
    )


def _status(value: object) -> FindingStatus:
    """One of the four finding statuses."""
    if value == "confirmed":
        return "confirmed"
    if value == "contradicted":
        return "contradicted"
    if value == "unverified":
        return "unverified"
    if value == "historical":
        return "historical"
    raise ValueError("status must be confirmed, contradicted, unverified, or historical")
