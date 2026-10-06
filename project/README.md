# CISC3024 AI Assignment #2: reduced I-EDL reproduction

This AI-generated implementation follows the saved **Protocol 1.0**, under
[`published_results/protocol_1.0.txt`](../published_results/protocol_1.0.txt). The protocol is unchanged. All nine final runs are complete and verified.
This is a reduced assignment reproduction, not the paper's full benchmark.
See the [root README](../README.md) for current installation, completed results,
and reproduction commands, including the audit/export helper.

## Sources and attribution

Developed with reference to Deng, Chen, Yu, Liu and Heng, **Uncertainty Estimation
by Fisher Information-based Evidential Deep Learning**, ICML 2023:
[paper](https://proceedings.mlr.press/v202/deng23b.html) and
[official repository](https://github.com/danruod/IEDL), which is MIT licensed
(copyright 2023 DanruoDeng). Inspected source commit:
`4aca7db2dabd4ccb46f72765360576e7dfe080d6`, particularly
`code_classical/models/FisherEvidentialN.py` and
`code_classical/configs/1_mnist/mnist-iedl.json`.
The mathematics was reimplemented; no large official source blocks were copied.
The attribution does not imply author endorsement or official benchmark fidelity.
The exact upstream MIT notice is preserved in [I-EDL-MIT.txt](../third_party_licenses/I-EDL-MIT.txt); see [ATTRIBUTION.md](../ATTRIBUTION.md) for provenance and licensing scope.

## Commands (PowerShell, from the assignment root)

```powershell
# Help only: never trains by default
& '.\.venv\Scripts\python.exe' '.\project\run.py'

# Bounded smoke verification, optionally downloading missing data
& '.\.venv\Scripts\python.exe' '.\project\run.py' smoke --download

# Fresh reproduction pilot: two full epochs per method, seed 999
& '.\.venv\Scripts\python.exe' '.\project\run.py' pilot

# Core final reproduction: matching pilot required; all seeds, then final tests
& '.\.venv\Scripts\python.exe' '.\project\run.py' full
```

Install the root requirements.txt before reproduction. All computation is CPU-only. Run a fresh
process so the 4 intra-op / 1 inter-op thread settings can be applied. Missing
datasets fail without `smoke --download`. Pilot and full output directories must
not already exist; this prevents accidental overwriting or silently repeating
runs. If a run fails, preserve its diagnostics and resolve it deliberately.
Checkpoint resume primitives exist in `training.py`; the CLI does not silently
resume partial experiments. Load only trusted locally generated `.pt` files.

## Modules

| File | Responsibility |
|---|---|
| config.py | Frozen constants, configuration/source fingerprint, timing rule |
| models.py | Shared 50,570-parameter CNN and exact paired copies |
| losses.py | Softmax CE, MSE-EDL, full I-EDL and diagnostic failures |
| data.py | Test-only Fashion downloads, saved stratified split, paired permutations |
| training.py | Independent Adam, weighted validation, stopping, checkpoint/RNG restore |
| evaluation.py | Predictive classification, OOD-positive AUROC/AP, selective risk |
| artifacts.py | JSON/CSV, seed summaries, plots and environment snapshots |
| run.py | Explicit smoke, pilot and full modes |
| smoke_test.py | Bounded verification without pilot/full/test-result evaluation |

## Objective and numerical fidelity

For evidential methods, `e=softplus(z)`, `alpha=e+1`, `S=sum(alpha)` and `q=alpha/S`.
With one-hot label `y`, let `v_k=alpha_k*(S-alpha_k)/(S^2*(S+1))` and
`a=alpha*(1-y)+y`. Define `KL=KL(Dir(a)||Dir(1))`.

* EDL: `sum_k[(y_k-q_k)^2+v_k] + anneal*KL`.
* I-EDL: `sum_k trigamma(alpha_k)*[(y_k-q_k)^2+v_k]`
  `-0.005*logdet(F) + anneal*KL`, with
  `F=diag(trigamma(alpha))-trigamma(S)*11^T`.
* Softmax: cross-entropy of raw logits; no evidential terms.

Losses average over examples. I-MSE uses `trigamma(alpha_k)`, rather than `F_kk`:
the rank-one term cancels under simplex constraints. A full 10x10 Fisher matrix
and differentiable Cholesky determine the log determinant. There is no detached
Fisher weighting, epsilon shift, jitter, clamping, or nonfinite replacement.
The CNN remains float32; logits are cast to float64 before every output/loss
transformation. Invalid values terminate the affected run with diagnostics.
I-EDL models a Dirichlet distribution over class probabilities; it does **not**
infer a Bayesian posterior over CNN weights.

## Operational conventions fixed before results

* Epoch index `t` is zero-based, consistent with the original training convention:
  the first epoch has `anneal=0`, then `0.1,...,1` at `t=10`.
  Histories also give completed epochs (`t+1`).
* NumPy PCG64(3024) permutes class indices in ascending class order. The first
  1,000 become training, the next 200 validation; class blocks are concatenated.
  All later shuffling comes from the shared CPU permutation seed `s+100000+t`.
* Patience counts consecutive non-improvements from the start. Stopping is gated
  until 15 completed epochs. Exact count/NLL ties keep the earlier checkpoint.
* The user explicitly confirmed **30 equal-width histogram bins on [0,1]**;
  the saved protocol had not specified their count. No training setting changed.
* Pilot epoch wall time includes the training pass and full validation, and
  excludes checkpoint and plot I/O. Only the slowest of six epoch times determines
  30/20/15 epochs at inclusive thresholds 160/240/320 seconds. Above 320, stop.
* `training_seconds` is the sum of training passes; `train_validation_seconds`
  also includes validation. Completed/selected epoch counts are reported.
* AURC is the arithmetic mean of risks for retained counts 1 through N (a
  right-endpoint discrete integral). Risk at c uses `ceil(c*N)` examples. Sort
  uncertainty increasingly, then original MNIST test index increasingly.
* ECE uses 15 equal-width bins `[b/15,(b+1)/15)`, with 1 in the final bin;
  weight absolute confidence/accuracy gaps by bin population / N. Empty bins
  contribute zero. Brier is class-summed, not divided by the class count.
* NLL uses stable log-softmax or `log(alpha)-log(S)`, without clipping q.
* OOD label is 1 and ID label is 0; larger uncertainty means more OOD-like.
  Primary score is `1-max(q)` for all methods. `10/S` is secondary and separately
  labelled for EDL/I-EDL. Mutual information and KMNIST are excluded.
* Exported Fashion-MNIST metadata may retain archival garment labels as `dataset_label`; digit-class labels are `-1` for OOD use. No Fashion-MNIST digit classification accuracy is computed.
* Final test predictions are generated only after all nine validation-selected
  fits finish, by reloading selected weights. Optimizer/RNG states are saved for
  reproducibility but not restored for deterministic final inference.

## Files generated by execution

* `datasets/`: official MNIST train/test files, **only Fashion-MNIST test files**,
  and `mnist_split_3024.npz`. Downloaded resources use torchvision MD5 checks.
  MNIST can use the standard CVDF mirror if torchvision's mirror is unavailable.
  A test-only FashionMNIST subclass avoids torchvision requiring training files.
* `evidence/verification/`: generated smoke verification records remain local. Only the two reproduction/export Python helpers are selected for publication.
* `results/smoke/`: configuration, a smoke checkpoint, and explicitly synthetic
  summary/plots. These are verification artifacts, never experimental results.
* `results/pilot/`: configuration, six epoch times, separate histories,
  runtime decision. No trained pilot weights are retained.
* `results/full/`: each seed/method's history, selected and last checkpoints,
  per-example probabilities/log probabilities/uncertainties, classification and
  OOD/rejection tables, mean ± sample SD (ddof=1), configuration and completion
  record. Plots include objective, validation accuracy/NLL, reliability,
  risk-coverage, MSP/primary ID-OOD histograms and evidential K/S histograms.

Selected public tables, plots, histories, split indices, frozen protocol, timing and verification records are in `published_results/`. Configuration/environment copies are sanitized; original generated outputs remain local.

Validation loss is example-weighted. Checkpoints include model, optimizer,
completed epoch, CPU RNGs, next permutation metadata and selection state.
The saved split is validated against regenerated indices and the target hash.
Pilot reuse requires the same source/configuration fingerprint and environment.

## Interpretation and deviations

Report observed differences and negative findings; do not claim universal
superiority, significance without appropriate analysis, or Bayesian weight
uncertainty. Do not tune after test/OOD results. Fixing incorrect mathematics,
metrics, shapes, paths or equivalent numerical calculations is debugging, not
permission to change hyperparameters. Any unavoidable design change must be
documented before dependent work as Protocol 1.1 or later, with old/new settings,
reason, date, affected runs and report disclosure. Preserve prior evidence.
