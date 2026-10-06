# CISC3024 AI Assignment #2: reduced I-EDL reproduction

A completed CPU-only comparison of Softmax, standard MSE-based Evidential Deep
Learning (EDL), and Fisher Information-based Evidential Deep Learning (I-EDL).
This implementation was AI-generated with Codex under student direction.
See [ATTRIBUTION.md](ATTRIBUTION.md) for paper, official-code and licensing notes.
This is a reduced assignment reproduction, not the paper's full benchmark.

## Frozen experiment

Protocol 1.0 uses a shared 50,570-parameter CNN, MNIST training/validation subsets
of 10,000/2,000 examples, split seed 3024, and paired training seeds 510/511/512.
Inputs use ToTensor only. Adam uses learning rate 0.001, with batch size 64.
CPU threads are 4 intra-op and 1 inter-op. The CNN is float32 and probability/loss
operations are float64. I-EDL's Fisher coefficient is 0.005. Checkpoints are
selected using validation only. Full MNIST and Fashion-MNIST test sets are used
after checkpoint restoration; Fashion-MNIST is OOD, not digit classification.

The runtime-only pilot's slowest epoch was 16.278363 seconds, selecting a
30-epoch maximum under the frozen thresholds. All nine final runs passed the
recorded verification. No pilot weights were reused or post-result tuning done.

## Selected completed results

Values are arithmetic mean +/- sample SD over three independent paired seeds.

| Method | Accuracy (%) | NLL | ECE (%) | Primary OOD AUROC |
|---|---:|---:|---:|---:|
| Softmax | 98.180 +/- 0.191 | 0.06784 +/- 0.00697 | 0.820 +/- 0.163 | 0.95065 +/- 0.00789 |
| EDL | 73.753 +/- 25.704 | 0.83162 +/- 0.68815 | 8.816 +/- 1.960 | 0.69098 +/- 0.29607 |
| I-EDL | 97.947 +/- 0.059 | 0.31730 +/- 0.02775 | 21.156 +/- 1.932 | 0.95487 +/- 0.00843 |

Primary uncertainty is 1-MSP; K/S is a separately reported evidential score.
I-EDL preserved high, stable accuracy but had worse calibration than Softmax.
Softmax had higher mean OOD average precision and lower mean AURC. EDL was
unstable across seeds. These results do not establish statistical significance,
universal superiority or Bayesian posterior inference over neural-network weights.

See [published_results/RESULTS.md](published_results/RESULTS.md) for complete
tables. Selected plots illustrate individual seeds; all-seed tables support
comparisons. Original results remain private and unchanged. Public configuration
and environment copies remove local paths; a provenance note explains this.

## Reproduction on a fresh checkout

Recorded environment: Python 3.11.0 on Windows 11, Intel Core i5-1340P, CPU only.
Exact dependency installation on a fresh machine has not been retested in this
publication preparation step. No CUDA, W&B or SEML is required.

```powershell
py -3.11 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' '.\project\run.py'             # help only
& '.\.venv\Scripts\python.exe' '.\project\run.py' smoke --download
& '.\.venv\Scripts\python.exe' '.\project\run.py' pilot
& '.\.venv\Scripts\python.exe' '.\evidence\verification\final_execution_audit.py' --execute
```

The final command explicitly starts all nine final runs and the observational
audit/export process. For core training/evaluation without those extra audit
exports, `project/run.py full` is the alternative; do not run both. The audit
helper exports secondary K/S risk curves and explicit per-example metadata from
saved predictions without additional inference. `write_final_report.py` in the
same directory exports the numerical RESULTS.md, not the university report.

Fresh reproduction must run its own pilot: stored public timings cannot bypass
the source/environment fingerprint check. The fingerprint includes checkout
paths, so it is not expected to match the historical execution fingerprint.
Individual source hashes are provided for byte-level source comparison.
Keep the project directory layout intact. Existing pilot/full output folders
cause the CLI to stop rather than overwrite results. Use a fresh checkout for
reproduction and load only trusted checkpoint files.

## Repository contents

- `project/*.py`: unchanged frozen implementation and bounded smoke verification.
- `published_results/`: compact tables, selected plots, protocol, split indices,
  runtime and verification records; no trained weights or dataset images.
- `evidence/verification/*.py`: only the two reproduction/export helpers.
- `requirements.txt`: six directly required packages, including SciPy for smoke tests.

The exact official MIT notice is preserved in [third_party_licenses/I-EDL-MIT.txt](third_party_licenses/I-EDL-MIT.txt). No blanket repository-wide license is asserted; see ATTRIBUTION.md.

Local virtual environments, downloaded datasets, raw prediction arrays,
checkpoints, personal reflection, prompts and internal workflow records are
excluded. [project/README.md](project/README.md) provides detailed implementation documentation; this root README provides installation, completed results and reproduction instructions.
