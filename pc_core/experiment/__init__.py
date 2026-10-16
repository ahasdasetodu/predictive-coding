"""Run named experiment specs and record their results."""

from pc_core.experiment.byproduct import ByproductSpec, train_byproduct
from pc_core.experiment.clue_bits import ClueBitsSpec, measure_clue_bits
from pc_core.experiment.findings import Finding, load_findings, parse_findings
from pc_core.experiment.mnist import ExperimentSpec, parse_cli, train_mnist, training_argv
from pc_core.experiment.runner import code_identity, run_spec
from pc_core.experiment.spec import TRAINERS, dump_spec, load_spec, parse_spec, spec_argv

__all__ = [
    "TRAINERS",
    "ByproductSpec",
    "ClueBitsSpec",
    "ExperimentSpec",
    "Finding",
    "code_identity",
    "dump_spec",
    "load_findings",
    "load_spec",
    "measure_clue_bits",
    "parse_cli",
    "parse_findings",
    "parse_spec",
    "run_spec",
    "spec_argv",
    "train_byproduct",
    "train_mnist",
    "training_argv",
]
