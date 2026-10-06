"""Readable tables from the verified, saved final results; no experiments."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results" / "full"
NAMES = {"softmax": "Softmax", "edl": "EDL", "iedl": "I-EDL"}


def read_csv(name):
    with (D / name).open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def format_number(value, scale=1.0):
    return f"{float(value) * scale:.6g}"


def generate():
    verification = json.loads((D / "final_verification.json").read_text(encoding="utf-8"))
    assert verification["status"] == "PASS" and len(verification["runs"]) == 9
    classification = read_csv("classification_metrics.csv")
    rejection = read_csv("ood_rejection_metrics.csv")
    secondary = read_csv("secondary_K_over_S_risk_metrics.csv")
    lines = ["# CISC3024 AI Assignment #2 — final Protocol 1.0 results", "",
             "All nine runs completed and passed the final verification. Maximum budget: 30 epochs; "
             "minimum before early stopping: 15; patience: 5. All runs used CPU only, the same frozen "
             "split, paired initial weights and paired minibatch orders. Selected validation checkpoints "
             "were reloaded before one MNIST-test and one Fashion-MNIST-OOD inference pass per run.", "",
             "The numbers below are individual-seed results or arithmetic mean ± sample SD (n=3, ddof=1). "
             "Predictions were not pooled or ensembled. Accuracy and ECE are shown as percentages; "
             "all risk, AURC, AUROC, and AP values remain fractions.", "",
             "## Run status and classification (MNIST test: 10,000 images)", ""]
    rows = []
    for r in classification:
        rows.append([r["seed"], NAMES[r["method"]], "PASS", r["completed_epochs"], r["selected_epoch"],
                     format_number(r["training_seconds"]), format_number(r["train_validation_seconds"]),
                     format_number(r["accuracy"], 100), format_number(r["nll"]),
                     format_number(r["brier"]), format_number(r["ece"], 100)])
    lines += [table(["Seed", "Method", "Status", "Epochs", "Selected", "Train s", "Train+val s",
                     "Accuracy %", "NLL", "Brier", "ECE %"], rows), "",
              "Training seconds exclude validation. Train+val seconds include every epoch's full "
              "validation pass. Plotting/checkpoint I/O are outside these per-epoch totals.", ""]

    def aggregate(name, methods, metrics):
        values = {(r["method"], r["metric"]): r for r in read_csv(name)}
        result = []
        for method in methods:
            row = [NAMES[method]]
            for key, _label, scale in metrics:
                value = values.get((method, key))
                row.append("—" if value is None else
                           format_number(value["mean"], scale) + " ± " + format_number(value["sample_sd"], scale))
            result.append(row)
        return table(["Method"] + [label for _, label, _ in metrics], result)

    lines += ["## Classification aggregate", "",
              aggregate("classification_summary.csv", ("softmax", "edl", "iedl"),
                        [("accuracy", "Accuracy %", 100), ("nll", "NLL", 1), ("brier", "Brier", 1),
                         ("ece", "ECE %", 100)]), "",
              "## Training aggregate", "",
              aggregate("classification_summary.csv", ("softmax", "edl", "iedl"),
                        [("completed_epochs", "Completed epochs", 1), ("selected_epoch", "Selected epoch", 1),
                         ("training_seconds", "Train s", 1), ("train_validation_seconds", "Train+val s", 1)]), "",
              "## OOD (Fashion-MNIST test: 10,000 images)", "",
              "MNIST is ID=0 and Fashion-MNIST is OOD=1 (positive). Primary uncertainty is 1−MSP; "
              "secondary evidential uncertainty is K/S. Fashion-MNIST garment labels are archival "
              "metadata only; no digit classification accuracy was calculated.", ""]
    fields = [("primary_ood_auroc", "Primary AUROC", 1), ("primary_ood_ap", "Primary AP", 1),
              ("evidential_ood_auroc", "K/S AUROC", 1), ("evidential_ood_ap", "K/S AP", 1)]
    rows = [[r["seed"], NAMES[r["method"]]] +
            [format_number(r[key]) if r.get(key) else "—" for key, _, _ in fields] for r in rejection]
    lines += [table(["Seed", "Method"] + [label for _, label, _ in fields], rows), "",
              aggregate("ood_rejection_summary.csv", ("softmax", "edl", "iedl"), fields), ""]
    risk_fields = [("aurc", "AURC", 1)] + [(f"risk_{c}", f"Risk @{c}%", 1) for c in (50, 80, 90, 100)]
    for title, rows, summary, methods in (
            ("Primary risk-coverage (1−MSP)", rejection, "ood_rejection_summary.csv", ("softmax", "edl", "iedl")),
            ("Secondary risk-coverage (K/S)", secondary, "secondary_K_over_S_risk_summary.csv", ("edl", "iedl"))):
        lines += ["## " + title, "",
                  table(["Seed", "Method"] + [label for _, label, _ in risk_fields],
                        [[r["seed"], NAMES[r["method"]]] + [format_number(r[key]) for key, _, _ in risk_fields] for r in rows]), "",
                  aggregate(summary, methods, risk_fields), ""]
    png_count = len(list(D.rglob("*.png")))
    lines += ["## Verification and artifacts", "",
              f"Verification: **PASS**. Execution wall time: {verification['execution_wall_seconds']:.3f} seconds "
              f"(includes training, final inference, checkpoint I/O and core plot generation). {png_count} PNG plots.", "",
              "No implementation bug corrections or protocol deviations occurred. Observational auditing "
              "did not change the frozen project source. Required K/S risk-coverage and explicit ID/OOD "
              "metadata exports used saved predictions only; no additional inference was run.", "",
              "Each seed/method directory contains history, training summary, selected.pt and last.pt, "
              "MNIST and Fashion prediction NPZs, metrics, per_example_records.npz, and plots. "
              "EDL/I-EDL also contain secondary K/S risk metrics and curve data. Histograms use exactly "
              "30 equal-width bins on [0,1]. Source hashes and the execution audit are retained.", "",
              "Per-example dataset_label stores each dataset's original label. mnist_digit_label is −1 "
              "for OOD; ood_status is 0/1, and original_index is the official index within that test set.", "",
              "These are descriptive results from a reduced experiment. No claim of universal superiority "
              "or statistical significance is made. I-EDL is evidential probability modeling, not a "
              "Bayesian posterior over CNN weights.", ""]
    (D / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Saved readable final tables: " + str(D / "RESULTS.md"))


if __name__ == "__main__":
    generate()
