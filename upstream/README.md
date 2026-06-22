# Upstream material (preserved verbatim)

This folder holds the upstream sources this project builds on, kept byte-identical to their
originals for attribution and provenance. Do not edit these files; any status notes about them
belong in this index, not in the files themselves.

## `README.authors.md`

The original README from the authors' repository for COSETTE and MARIUS (Simon Lepage,
Jérémie Mary, David Picard; CRITEO AI Lab and ENPC).

- Source: https://github.com/Simon-Lepage/cosette_and_marius
- Paper: https://arxiv.org/abs/2508.14910

## `REPRODUCIBILITY.md`

The course head TA's original reproducibility study (single seed, on Beauty, Video Games, and
Arts and Crafts), taken from the head TA's fork.

- Source: https://github.com/2t2c/cosette_and_marius

This study is superseded by our reproduction in
[`../REPLICATION_REPORT.md`](../REPLICATION_REPORT.md), which uses five seeds on Beauty and
Sports, adds the root-cause analysis of the gap, and includes the paper-faithful SASRec++
re-run. Where the two disagree, trust the report and the committed `reports/` artifacts. The
TA's document is kept here unchanged for attribution.
