"""Report output, with synthetic smoke artifacts kept separate from results."""
import csv
import json
import os
from pathlib import Path
from importlib.metadata import version
import numpy as np
from config import P, ROOT, snapshot, fingerprint

# Keep Matplotlib's cache inside the authorized project, and use no GUI.
os.environ["MPLCONFIGDIR"] = str(ROOT / ".venv" / "matplotlib-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluation import calibration, risk_coverage


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")


def environment():
    import platform
    import sys
    import torch
    return {"python": platform.python_version(), "interpreter": sys.executable,
            "os": platform.platform(), "cpu_count": os.cpu_count(),
            "packages": {p: version(p) for p in
                         ("torch", "torchvision", "numpy", "matplotlib", "scikit-learn")},
            "threads": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
            "torch_cuda_available": torch.cuda.is_available()}


def write_snapshot(directory):
    write_json(Path(directory) / "configuration.json", {
        **snapshot(), "source_fingerprint": fingerprint(), "environment": environment()})


def write_table(path, rows):
    if not rows:
        raise ValueError("No rows to report")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_table(path, rows):
    """Mean and sample SD across the three predeclared seeds, ddof=1."""
    summary = []
    for method in sorted({r["method"] for r in rows}):
        group = [r for r in rows if r["method"] == method]
        if sorted(r["seed"] for r in group) != list(P.seeds):
            raise ValueError("Summaries require all three frozen seeds")
        for metric in group[0]:
            if metric in ("method", "seed"):
                continue
            values = np.asarray([r[metric] for r in group], dtype=np.float64)
            mean, sd = values.mean(), values.std(ddof=P.sample_std_ddof)
            summary.append({"method": method, "metric": metric, "mean": float(mean),
                            "sample_sd": float(sd), "mean_plus_minus_sd": f"{mean:.6g} ± {sd:.6g}"})
    write_table(path, summary)


def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history, directory, title):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    epochs = [r["completed_epochs"] for r in history]
    fig, ax = plt.subplots()
    ax.plot(epochs, [r["train"]["objective"] for r in history], label="train")
    ax.plot(epochs, [r["validation_loss"]["objective"] for r in history], label="validation")
    ax.set(xlabel="Completed epoch", ylabel="Method-specific objective", title=title)
    ax.legend()
    save_plot(fig, directory / "objective.png")
    for metric in ("accuracy", "nll"):
        fig, ax = plt.subplots()
        ax.plot(epochs, [r["validation"][metric] for r in history])
        ax.set(xlabel="Completed epoch", ylabel=f"Validation {metric}", title=title)
        save_plot(fig, directory / f"validation_{metric}.png")


def plot_predictions(id_record, ood_record, directory, title):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    q, y = id_record["probabilities"], id_record["labels"]
    cal = calibration(q, y)
    mask = cal["count"] > 0
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.plot(cal["confidence"][mask], cal["accuracy"][mask], "o-")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence",
           ylabel="Fraction correct", title=title + " reliability (15 bins)")
    save_plot(fig, directory / "reliability.png")
    curve = risk_coverage(q, y, id_record["primary_uncertainty"], id_record["index"])
    fig, ax = plt.subplots()
    ax.plot(curve["coverage"], curve["risk"])
    ax.set(xlabel="Coverage", ylabel="Risk (error fraction)", title=title)
    save_plot(fig, directory / "risk_coverage.png")
    bins = np.linspace(*P.histogram_range, P.histogram_bins + 1)
    scores = {"msp": (q.max(1), ood_record["probabilities"].max(1)),
              "primary_uncertainty": (id_record["primary_uncertainty"],
                                      ood_record["primary_uncertainty"])}
    if "evidential_uncertainty" in id_record:
        scores["K_over_S"] = (id_record["evidential_uncertainty"],
                              ood_record["evidential_uncertainty"])
    for name, (id_scores, ood_scores) in scores.items():
        fig, ax = plt.subplots()
        ax.hist(id_scores, bins=bins, alpha=0.5, density=True, label="MNIST ID")
        ax.hist(ood_scores, bins=bins, alpha=0.5, density=True, label="Fashion-MNIST OOD")
        ax.set(xlabel=name, ylabel="Density", title=title)
        ax.legend()
        save_plot(fig, directory / f"{name}_histogram.png")
    return curve
