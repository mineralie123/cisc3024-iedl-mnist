# CISC3024 AI Assignment #2 — final Protocol 1.0 results

All nine runs completed and passed the final verification. Maximum budget: 30 epochs; minimum before early stopping: 15; patience: 5. All runs used CPU only, the same frozen split, paired initial weights and paired minibatch orders. Selected validation checkpoints were reloaded before one MNIST-test and one Fashion-MNIST-OOD inference pass per run.

The numbers below are individual-seed results or arithmetic mean ± sample SD (n=3, ddof=1). Predictions were not pooled or ensembled. Accuracy and ECE are shown as percentages; all risk, AURC, AUROC, and AP values remain fractions.

## Run status and classification (MNIST test: 10,000 images)

| Seed | Method | Status | Epochs | Selected | Train s | Train+val s | Accuracy % | NLL | Brier | ECE % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 510 | Softmax | PASS | 22 | 17 | 299.655 | 330.967 | 98 | 0.0751548 | 0.0303593 | 1.00876 |
| 510 | EDL | PASS | 28 | 23 | 484.674 | 526.601 | 97.25 | 0.207408 | 0.0591188 | 8.91352 |
| 510 | I-EDL | PASS | 30 | 25 | 461.943 | 506.566 | 97.99 | 0.326982 | 0.0921624 | 22.0246 |
| 511 | Softmax | PASS | 23 | 18 | 715.22 | 745.501 | 98.38 | 0.0612807 | 0.0265715 | 0.726642 |
| 511 | EDL | PASS | 18 | 13 | 328.034 | 353.01 | 77.71 | 0.71791 | 0.259559 | 10.7253 |
| 511 | I-EDL | PASS | 28 | 23 | 421.603 | 463.101 | 97.97 | 0.286008 | 0.0773926 | 18.9424 |
| 512 | Softmax | PASS | 20 | 15 | 266.397 | 294.301 | 98.16 | 0.0670939 | 0.0286509 | 0.726038 |
| 512 | EDL | PASS | 15 | 1 | 210.699 | 232.249 | 46.3 | 1.56954 | 0.668738 | 6.80821 |
| 512 | I-EDL | PASS | 25 | 20 | 375.08 | 412.218 | 97.88 | 0.338919 | 0.0979603 | 22.5009 |

Training seconds exclude validation. Train+val seconds include every epoch's full validation pass. Plotting/checkpoint I/O are outside these per-epoch totals.

## Classification aggregate

| Method | Accuracy % | NLL | Brier | ECE % |
| --- | --- | --- | --- | --- |
| Softmax | 98.18 ± 0.190788 | 0.0678431 ± 0.00696734 | 0.0285272 ± 0.00189695 | 0.820479 ± 0.163054 |
| EDL | 73.7533 ± 25.7044 | 0.831621 ± 0.688151 | 0.329139 ± 0.310709 | 8.81568 ± 1.96038 |
| I-EDL | 97.9467 ± 0.0585947 | 0.317303 ± 0.0277513 | 0.0891718 ± 0.010605 | 21.156 ± 1.93173 |

## Training aggregate

| Method | Completed epochs | Selected epoch | Train s | Train+val s |
| --- | --- | --- | --- | --- |
| Softmax | 21.6667 ± 1.52753 | 16.6667 ± 1.52753 | 427.091 ± 250.081 | 456.923 ± 250.587 |
| EDL | 20.3333 ± 6.80686 | 12.3333 ± 11.0151 | 341.136 ± 137.457 | 370.62 ± 147.964 |
| I-EDL | 27.6667 ± 2.51661 | 22.6667 ± 2.51661 | 419.542 ± 43.4681 | 460.628 ± 47.2228 |

## OOD (Fashion-MNIST test: 10,000 images)

MNIST is ID=0 and Fashion-MNIST is OOD=1 (positive). Primary uncertainty is 1−MSP; secondary evidential uncertainty is K/S. Fashion-MNIST garment labels are archival metadata only; no digit classification accuracy was calculated.

| Seed | Method | Primary AUROC | Primary AP | K/S AUROC | K/S AP |
| --- | --- | --- | --- | --- | --- |
| 510 | Softmax | 0.957723 | 0.952194 | — | — |
| 510 | EDL | 0.940744 | 0.90366 | 0.940578 | 0.903672 |
| 510 | I-EDL | 0.964475 | 0.933019 | 0.964624 | 0.932211 |
| 511 | Softmax | 0.952086 | 0.935866 | — | — |
| 511 | EDL | 0.768279 | 0.617507 | 0.768649 | 0.618078 |
| 511 | I-EDL | 0.95142 | 0.931585 | 0.952282 | 0.932466 |
| 512 | Softmax | 0.942139 | 0.920849 | — | — |
| 512 | EDL | 0.363929 | 0.394633 | 0.31592 | 0.402689 |
| 512 | I-EDL | 0.948705 | 0.904468 | 0.949291 | 0.904894 |

| Method | Primary AUROC | Primary AP | K/S AUROC | K/S AP |
| --- | --- | --- | --- | --- |
| Softmax | 0.950649 ± 0.00789076 | 0.936303 ± 0.0156775 | — | — |
| EDL | 0.690984 ± 0.296074 | 0.6386 ± 0.255168 | 0.675049 ± 0.322676 | 0.64148 ± 0.25131 |
| I-EDL | 0.954866 ± 0.00843102 | 0.923024 ± 0.0160859 | 0.955399 ± 0.00812813 | 0.92319 ± 0.0158457 |

## Primary risk-coverage (1−MSP)

| Seed | Method | AURC | Risk @50% | Risk @80% | Risk @90% | Risk @100% |
| --- | --- | --- | --- | --- | --- | --- |
| 510 | Softmax | 0.000829984 | 0 | 0.00125 | 0.00144444 | 0.02 |
| 510 | EDL | 0.00146395 | 0.0002 | 0.00125 | 0.00355556 | 0.0275 |
| 510 | I-EDL | 0.000882101 | 0 | 0.000625 | 0.002 | 0.0201 |
| 511 | Softmax | 0.000581736 | 0 | 0.000125 | 0.000777778 | 0.0162 |
| 511 | EDL | 0.028858 | 0.0004 | 0.041625 | 0.137667 | 0.2229 |
| 511 | I-EDL | 0.000794377 | 0 | 0.0005 | 0.00155556 | 0.0203 |
| 512 | Softmax | 0.000731842 | 0 | 0.0005 | 0.00155556 | 0.0184 |
| 512 | EDL | 0.314904 | 0.3682 | 0.47075 | 0.502556 | 0.537 |
| 512 | I-EDL | 0.00113 | 0.0002 | 0.00075 | 0.00255556 | 0.0212 |

| Method | AURC | Risk @50% | Risk @80% | Risk @90% | Risk @100% |
| --- | --- | --- | --- | --- | --- |
| Softmax | 0.00071452 ± 0.000125027 | 0 ± 0 | 0.000625 ± 0.000572822 | 0.00125926 ± 0.00042066 | 0.0182 ± 0.00190788 |
| EDL | 0.115075 ± 0.173598 | 0.122933 ± 0.212407 | 0.171208 ± 0.260195 | 0.214593 ± 0.258241 | 0.262467 ± 0.257044 |
| I-EDL | 0.000935494 ± 0.000174067 | 6.66667e-05 ± 0.00011547 | 0.000625 ± 0.000125 | 0.00203704 ± 0.000501028 | 0.0205333 ± 0.000585947 |

## Secondary risk-coverage (K/S)

| Seed | Method | AURC | Risk @50% | Risk @80% | Risk @90% | Risk @100% |
| --- | --- | --- | --- | --- | --- | --- |
| 510 | EDL | 0.00147168 | 0.0002 | 0.00125 | 0.00355556 | 0.0275 |
| 510 | I-EDL | 0.000907163 | 0 | 0.000625 | 0.00211111 | 0.0201 |
| 511 | EDL | 0.0289058 | 0.0004 | 0.041625 | 0.138 | 0.2229 |
| 511 | I-EDL | 0.000823761 | 0 | 0.0005 | 0.00144444 | 0.0203 |
| 512 | EDL | 0.479399 | 0.5262 | 0.535875 | 0.537 | 0.537 |
| 512 | I-EDL | 0.00118339 | 0.0002 | 0.000875 | 0.00288889 | 0.0212 |

| Method | AURC | Risk @50% | Risk @80% | Risk @90% | Risk @100% |
| --- | --- | --- | --- | --- | --- |
| EDL | 0.169925 ± 0.268363 | 0.1756 ± 0.303629 | 0.192917 ± 0.297696 | 0.226185 ± 0.27744 | 0.262467 ± 0.257044 |
| I-EDL | 0.000971438 ± 0.000188234 | 6.66667e-05 ± 0.00011547 | 0.000666667 ± 0.000190941 | 0.00214815 ± 0.000722934 | 0.0205333 ± 0.000585947 |

## Verification and artifacts

Verification: **PASS**. Execution wall time: 4035.700 seconds (includes training, final inference, checkpoint I/O and core plot generation). 75 PNG plots.

No implementation bug corrections or protocol deviations occurred. Observational auditing did not change the frozen project source. Required K/S risk-coverage and explicit ID/OOD metadata exports used saved predictions only; no additional inference was run.

Each seed/method directory contains history, training summary, selected.pt and last.pt, MNIST and Fashion prediction NPZs, metrics, per_example_records.npz, and plots. EDL/I-EDL also contain secondary K/S risk metrics and curve data. Histograms use exactly 30 equal-width bins on [0,1]. Source hashes and the execution audit are retained.

Per-example dataset_label stores each dataset's original label. mnist_digit_label is −1 for OOD; ood_status is 0/1, and original_index is the official index within that test set.

These are descriptive results from a reduced experiment. No claim of universal superiority or statistical significance is made. I-EDL is evidential probability modeling, not a Bayesian posterior over CNN weights.
