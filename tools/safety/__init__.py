"""AstraCore Neo — functional-safety tooling.

This package implements the analyses required by ISO 26262-5 §7.4.5
(quantitative analysis / FMEDA) and ISO 26262-11 §4.6 (semiconductor
guidance). It is intentionally separate from ``tools/npu_ref/`` (the
compute reference) because functional-safety tooling has a different
release cadence (driven by safety-case revisions) and different
qualification requirements (Tool Confidence Level evaluations per
ISO 26262-8 §11).

Modules
-------
``fmeda``
    Failure Modes, Effects, and Diagnostic Analysis. Reads a per-module
    failure-mode catalog (YAML) plus a safety-mechanism table (YAML)
    and produces a markdown FMEDA report with SPFM / LFM / PMHF
    aggregates. Used by the safety case to derive the quantitative
    metrics ISO 26262 requires.
"""
