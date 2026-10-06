# Attribution and scope

The assignment implementation was AI-generated using Codex under student direction.
AI assistance covered algorithm search, paper/code inspection, implementation,
verification, execution and analysis under a student-directed frozen protocol.

The principal reference is Deng, Chen, Yu, Liu and Heng, *Uncertainty Estimation
by Fisher Information-based Evidential Deep Learning*, ICML 2023:
https://proceedings.mlr.press/v202/deng23b.html

Official implementation: https://github.com/danruod/IEDL
Reference commit: 4aca7db2dabd4ccb46f72765360576e7dfe080d6.
The official repository was used as a mathematical and implementation reference.
Its exact MIT license is preserved in
[third_party_licenses/I-EDL-MIT.txt](third_party_licenses/I-EDL-MIT.txt), with
`Copyright (c) 2023 DanruoDeng`. The 1,067-byte file was retrieved without
reconstructing or editing its text from LICENSE at the reference commit above.
Verified Git blob SHA-1: `c22ebc810ee2690c189ae137b2e6d78469838984`.
The same license blob was present on the current main branch when checked.

This is a reduced CPU-only assignment reproduction, not the original benchmark
or an official implementation. Attribution does not imply author endorsement.
Standard EDL is included as a baseline; its original paper should also be cited
in the assignment report. MNIST and Fashion-MNIST are downloaded separately;
dataset files and third-party package sources are not redistributed here.

The upstream license requires its copyright and permission notices to accompany
copies or substantial portions of that software. The full notice is preserved
here conservatively; it does not grant a license to the entire assignment.
No separate NOTICE, COPYING or AUTHORS file was found in the inspected upstream
repository trees. Its README requests citation and acknowledges the projects below.
The official repository acknowledges Posterior Network and Firth Bias Reduction
in Few-shot Distribution Calibration; this assignment does not implement the
official few-shot pipeline.

## Conservative provenance review

Implementation history records a clean mathematical reimplementation. Review of
local losses.py against code_classical/models/FisherEvidentialN.py supports that
account: evidence construction, EDL terms, Fisher weighting and masked KL follow
the same mathematics, but the local implementation separates these into functions,
uses float64 transformations, and computes a full Fisher matrix with Cholesky
rather than the upstream determinant expression. The local loss has no upstream
CUDA initialization or nonfinite-value replacement. No substantial verbatim
upstream block was identified in this targeted review.

Explicit mathematical/implementation attribution applies especially to
project/losses.py, and architecture/configuration inspiration to project/models.py
and project/config.py. The reduced architecture and settings follow the frozen
student protocol, not an unchanged upstream configuration. This review is not an
exhaustive code-similarity audit or a legal guarantee. Preserve applicable notices
if additional copied/adapted material is identified.

No repository-wide license is granted by this preparation step. The upstream
MIT license must not be presented as automatically licensing the assignment's
own code, documentation, datasets or other dependencies. A repository-wide
license choice is deferred until provenance and ownership have been reviewed.
