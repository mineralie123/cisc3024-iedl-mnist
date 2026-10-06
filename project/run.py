"""Explicit smoke / pilot / full commands. No argument means help only."""
import argparse
import json
from pathlib import Path
import numpy as np
from config import P, METHODS, RESULTS_DIR, fingerprint, timing_budget
from models import ReducedCNN, paired_models
from data import load_datasets, loader
from training import configure_cpu, seed_all, fit, restore_checkpoint
from evaluation import collect_predictions, evaluate_id, ood_metrics
from artifacts import (write_json, write_snapshot, write_table, summary_table,
                       plot_predictions, environment)


def run_pilot():
    directory = RESULTS_DIR / "pilot"
    directory.mkdir(parents=True, exist_ok=False)
    write_snapshot(directory)
    datasets = load_datasets(download=False)
    seed_all(P.pilot_seed)
    models = paired_models(P.pilot_seed)
    timings = {}
    for method in METHODS:
        result = fit(models[method], method, P.pilot_seed, datasets, P.pilot_epochs)
        timings[method] = [r["epoch_seconds"] for r in result["history"]]
        write_json(directory / f"{method}_history.json", result)
    slowest = max(seconds for values in timings.values() for seconds in values)
    budget = timing_budget(slowest)
    record = {"protocol": P.version, "seed": P.pilot_seed,
              "source_fingerprint": fingerprint(), "environment": environment(),
              "epoch_seconds": timings, "slowest_epoch_seconds": slowest,
              "maximum_epochs": budget, "practical": budget is not None,
              "timing_scope": "complete training epoch plus validation; excludes checkpoint/plot I/O",
              "selection_basis": "slowest epoch runtime only; no accuracy/uncertainty input"}
    write_json(directory / "timing.json", record)
    del models  # No pilot checkpoint survives or enters final training.
    if budget is None:
        print("STOP: Protocol 1.0 is impractical (>320 s). No final training started.")
    else:
        print(f"Pilot only complete. Frozen final maximum: {budget} epochs. Run full separately.")


def approved_budget():
    path = RESULTS_DIR / "pilot" / "timing.json"
    if not path.exists():
        raise RuntimeError("Run the separate timing pilot first; full training has no fallback budget")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["source_fingerprint"] != fingerprint() or record["protocol"] != P.version:
        raise RuntimeError("Source/configuration differs from pilot; do not reuse its timing decision")
    if record["seed"] != P.pilot_seed or record["environment"] != environment():
        raise RuntimeError("Pilot seed or execution environment differs")
    if set(record["epoch_seconds"]) != set(METHODS):
        raise RuntimeError("Pilot must contain all methods")
    values = []
    for method in METHODS:
        times = record["epoch_seconds"][method]
        if len(times) != P.pilot_epochs:
            raise RuntimeError("Pilot must contain two complete epochs per method")
        for seconds in times:
            timing_budget(seconds)  # Validate positive finite durations.
        values.extend(times)
    slowest = max(values)
    budget = timing_budget(slowest)
    if slowest != record["slowest_epoch_seconds"] or budget != record["maximum_epochs"]:
        raise RuntimeError("Pilot budget record is inconsistent")
    if budget is None:
        raise RuntimeError("Protocol impractical: final training is prohibited by timing rule")
    return budget


def run_full():
    max_epochs = approved_budget()
    datasets = load_datasets(download=False)
    directory = RESULTS_DIR / "full"
    directory.mkdir(parents=True, exist_ok=False)
    write_snapshot(directory)
    trained = []
    # Complete all validation-selected fits before accessing any test predictions.
    for seed in P.seeds:
        seed_all(seed)
        models = paired_models(seed)
        for method in METHODS:
            run_dir = directory / f"seed_{seed}" / method
            result = fit(models[method], method, seed, datasets, max_epochs, run_dir)
            write_json(run_dir / "training_summary.json", {k: v for k, v in result.items() if k != "history"})
            trained.append((seed, method, run_dir, result))
        del models
    classification_rows, rejection_rows = [], []
    for seed, method, run_dir, result in trained:
        model = ReducedCNN()
        state = restore_checkpoint(run_dir / "selected.pt", model)
        id_record, _ = collect_predictions(model, method, loader(datasets.test))
        ood_record, _ = collect_predictions(model, method, loader(datasets.ood), labelled=False)
        np.savez_compressed(run_dir / "mnist_test_predictions.npz", **id_record)
        np.savez_compressed(run_dir / "fashion_ood_predictions.npz", **ood_record)
        metrics, curve = evaluate_id(id_record)
        classification_rows.append({"method": method, "seed": seed, **metrics,
                                    "training_seconds": result["training_seconds"],
                                    "train_validation_seconds": result["train_validation_seconds"],
                                    "completed_epochs": result["completed_epochs"],
                                    "selected_epoch": state["epoch"]})
        primary = ood_metrics(id_record["primary_uncertainty"], ood_record["primary_uncertainty"])
        rejection = {"method": method, "seed": seed,
                     "primary_ood_auroc": primary["auroc"],
                     "primary_ood_ap": primary["average_precision"], **curve["metrics"]}
        if method != "softmax":
            secondary = ood_metrics(id_record["evidential_uncertainty"],
                                    ood_record["evidential_uncertainty"])
            rejection.update(evidential_ood_auroc=secondary["auroc"],
                             evidential_ood_ap=secondary["average_precision"])
        rejection_rows.append(rejection)
        plot_predictions(id_record, ood_record, run_dir / "plots", f"{method}, seed {seed}")
        np.savez_compressed(run_dir / "risk_coverage.npz", coverage=curve["coverage"],
                            risk=curve["risk"], order=curve["order"])
        write_json(run_dir / "metrics.json", {"classification": classification_rows[-1],
                                              "ood_and_rejection": rejection})
    write_table(directory / "classification_metrics.csv", classification_rows)
    write_table(directory / "ood_rejection_metrics.csv", rejection_rows)
    summary_table(directory / "classification_summary.csv", classification_rows)
    summary_table(directory / "ood_rejection_summary.csv", rejection_rows)
    write_json(directory / "completion.json", {"status": "complete", "runs": len(trained),
                                               "source_fingerprint": fingerprint(),
                                               "maximum_epochs": max_epochs})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("smoke", "pilot", "full"))
    parser.add_argument("--download", action="store_true", help="Dataset downloads allowed in smoke mode only")
    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        return
    if args.download and args.mode != "smoke":
        parser.error("--download is only allowed with smoke")
    configure_cpu()
    if args.mode == "smoke":
        from smoke_test import run_smoke
        run_smoke(download=args.download)
    elif args.mode == "pilot":
        run_pilot()
    else:
        run_full()


if __name__ == "__main__":
    main()
