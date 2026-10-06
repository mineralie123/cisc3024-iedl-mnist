"""Observational final-run audit and required exports from saved predictions.

The Protocol 1.0 project source is untouched. Wrappers only check/hash objects,
then invoke the existing functions with their original arguments. No extra
training or inference is performed. Explicit --execute is required to train.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project"))
import numpy as np
import torch
import run
import training
from config import P, METHODS, RESULTS_DIR, SPLIT_PATH, fingerprint, snapshot
from models import ReducedCNN
from data import epoch_order
from evaluation import classification_metrics, ood_metrics, risk_coverage
from artifacts import (write_json, write_table, summary_table, environment,
                       plt, save_plot)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    return hashlib.sha256(np.asarray(array, dtype=np.int64).tobytes()).hexdigest()


def model_hash(model_or_state):
    state = model_or_state.state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_finite(value, name="value"):
    if isinstance(value, dict):
        for k, v in value.items():
            verify_finite(v, name + "." + k)
    elif isinstance(value, (list, tuple)):
        for v in value:
            verify_finite(v, name)
    elif isinstance(value, (float, int, np.ndarray)):
        assert np.isfinite(value).all(), name


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()

    def flush(self):
        self.stream.flush()
        self.log.flush()


def execute():
    training.configure_cpu()
    assert run.approved_budget() == 30
    assert not (RESULTS_DIR / "full").exists(), "Do not overwrite or silently resume final runs"
    sources = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT / "project").glob("*.py"))}
    document = ROOT / "evidence" / "summaries" / "04_frozen_protocol_summary.txt"
    audit = {"status": "running", "protocol": P.version, "maximum_epochs": 30,
             "source_fingerprint": fingerprint(), "core_source_hashes": sources,
             "protocol_document_sha256": file_hash(document), "split_file_sha256": file_hash(SPLIT_PATH),
             "environment": environment(), "initial_weights": [], "epoch_orders": [],
             "checkpoint_restorations": [], "final_evaluations": [],
             "implementation_changes": [], "protocol_deviations": []}
    audit_path = RESULTS_DIR / "final_execution_audit.json"
    write_json(audit_path, audit)
    original_fit = run.fit
    original_loader = training.loader
    original_restore = run.restore_checkpoint
    original_collect = run.collect_predictions
    active = {}
    selected = {}
    initial_hashes = {}
    order_hashes = {}
    data_refs = {}

    def audited_fit(model, method, seed, datasets, max_epochs, directory=None):
        assert seed in P.seeds and method in METHODS and max_epochs == 30
        assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters())
        assert sum(p.numel() for p in model.parameters()) == 50570
        # Independently regenerate only initial weights, without changing RNGs.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            expected_hash = model_hash(ReducedCNN())
        initial = model_hash(model)
        assert initial == expected_hash
        if seed in initial_hashes:
            assert initial_hashes[seed] == initial
        initial_hashes[seed] = initial
        with np.load(SPLIT_PATH, allow_pickle=False) as split:
            assert np.array_equal(datasets.train_indices, split["train"])
            assert np.array_equal(datasets.validation_indices, split["validation"])
        assert np.bincount(datasets.train.dataset.targets[datasets.train_indices].numpy()).tolist() == [1000] * 10
        assert np.bincount(datasets.validation.dataset.targets[datasets.validation_indices].numpy()).tolist() == [200] * 10
        assert not np.intersect1d(datasets.train_indices, datasets.validation_indices).size
        data_refs.update(test=datasets.test, ood=datasets.ood)
        active.update(seed=seed, method=method, train=datasets.train, validation=datasets.validation)
        audit["initial_weights"].append({"seed": seed, "method": method, "sha256": initial,
                                         "matches_fresh_seed_initialization": True,
                                         "train_split_sha256": array_hash(datasets.train.indices),
                                         "validation_split_sha256": array_hash(datasets.validation.indices)})
        write_json(audit_path, audit)
        result = original_fit(model, method, seed, datasets, max_epochs, directory)
        audit.setdefault("completed_runs", []).append({"seed": seed, "method": method,
                                                       "epochs": result["completed_epochs"]})
        write_json(audit_path, audit)
        return result

    def audited_loader(dataset, seed=None, epoch=None):
        batches = original_loader(dataset, seed, epoch)
        assert batches.num_workers == 0 and not batches.pin_memory and not batches.drop_last
        if seed is not None:
            assert dataset is active["train"] and seed == active["seed"]
            assert len(dataset) == 10000 and batches.batch_size == 64
            actual = np.asarray(batches.sampler, dtype=np.int64)
            expected = epoch_order(seed, epoch).numpy()
            assert np.array_equal(actual, expected)
            digest = array_hash(actual)
            key = (seed, epoch)
            if key in order_hashes:
                assert order_hashes[key] == digest
            order_hashes[key] = digest
            audit["epoch_orders"].append({"seed": seed, "method": active["method"],
                                          "zero_based_epoch": epoch, "generator_seed": seed + 100000 + epoch,
                                          "permutation_sha256": digest, "examples": 10000})
        else:
            assert dataset is active["validation"] and len(dataset) == 2000
            assert batches.batch_size == 256
        return batches

    def audited_restore(path, model, *args, **kwargs):
        state = original_restore(path, model, *args, **kwargs)
        seed, method = state["seed"], state["method"]
        assert seed in P.seeds and method in METHODS
        digest = model_hash(model)
        assert digest == model_hash(state["model"])
        selected[id(model)] = (seed, method, digest)
        audit["checkpoint_restorations"].append({"seed": seed, "method": method,
                                                  "epoch": state["epoch"], "model_sha256": digest,
                                                  "checkpoint": str(Path(path).relative_to(ROOT))})
        write_json(audit_path, audit)
        return state

    def audited_collect(model, method, batches, *, labelled=True, epoch=None):
        assert len(audit["completed_runs"]) == 9, "All fits must precede final evaluation"
        seed, restored_method, digest = selected[id(model)]
        assert method == restored_method and model_hash(model) == digest and epoch is None
        kind = "mnist_test" if labelled else "fashion_ood"
        assert batches.dataset is data_refs["test" if labelled else "ood"]
        assert len(batches.dataset) == 10000
        assert not any(e["seed"] == seed and e["method"] == method and e["dataset"] == kind
                       for e in audit["final_evaluations"]), "Do not repeat final inference"
        result = original_collect(model, method, batches, labelled=labelled, epoch=epoch)
        audit["final_evaluations"].append({"seed": seed, "method": method, "dataset": kind,
                                           "examples": 10000, "restored_model_sha256": digest})
        write_json(audit_path, audit)
        return result

    run.fit = audited_fit
    training.loader = audited_loader
    run.restore_checkpoint = audited_restore
    run.collect_predictions = audited_collect
    start = time.perf_counter()
    log_path = RESULTS_DIR / "final_execution.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            old_stdout = sys.stdout
            sys.stdout = Tee(old_stdout, log)
            try:
                run.run_full()
            finally:
                sys.stdout = old_stdout
        audit["execution_wall_seconds"] = time.perf_counter() - start
        audit["status"] = "experiment_complete"
        assert sources == {str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT / "project").glob("*.py"))}
        assert audit["protocol_document_sha256"] == file_hash(document)
        assert audit["split_file_sha256"] == file_hash(SPLIT_PATH)
        write_json(audit_path, audit)
        finish(audit)
    except Exception:
        audit["status"] = "failed"
        audit["error"] = traceback.format_exc()
        write_json(audit_path, audit)
        raise
    finally:
        run.fit, training.loader = original_fit, original_loader
        run.restore_checkpoint, run.collect_predictions = original_restore, original_collect


def finish(audit):
    """Verify saved outputs and export only the explicitly requested analyses."""
    directory = RESULTS_DIR / "full"
    config = json.loads((directory / "configuration.json").read_text())
    assert config["source_fingerprint"] == audit["source_fingerprint"] == fingerprint()
    assert config["protocol"] == json.loads(json.dumps(snapshot()["protocol"]))
    intended = {(seed, method) for seed in P.seeds for method in METHODS}
    assert {(r["seed"], r["method"]) for r in audit["completed_runs"]} == intended
    assert len(audit["initial_weights"]) == len(audit["checkpoint_restorations"]) == 9
    assert len(audit["final_evaluations"]) == 18
    with (directory / "classification_metrics.csv").open(encoding="utf-8") as f:
        classification_rows = list(csv.DictReader(f))
    with (directory / "ood_rejection_metrics.csv").open(encoding="utf-8") as f:
        primary_rows = list(csv.DictReader(f))
    assert len(classification_rows) == len(primary_rows) == 9
    # Labels loaded for archival metadata only; no model evaluation is performed.
    from data import load_datasets
    datasets = load_datasets(download=False)
    secondary_rows, run_statuses = [], []
    for seed in P.seeds:
        paired = [r["sha256"] for r in audit["initial_weights"] if r["seed"] == seed]
        assert len(paired) == 3 and len(set(paired)) == 1
        for method in METHODS:
            run_dir = directory / f"seed_{seed}" / method
            history = json.loads((run_dir / "history.json").read_text())
            summary = json.loads((run_dir / "training_summary.json").read_text())
            metrics = json.loads((run_dir / "metrics.json").read_text())
            verify_finite(history); verify_finite(summary); verify_finite(metrics)
            n = len(history)
            assert 15 <= n <= 30 and summary["completed_epochs"] == n
            selector = training.new_selection()
            for i, row in enumerate(history):
                assert row["completed_epochs"] == i + 1 and row["loss_epoch"] == i
                assert row["validation"]["examples"] == 2000
                assert row["kl_annealing"] == min(1, i / 10)
                assert training.select_checkpoint(selector, row["validation"], i + 1) == row["selected"]
                if i < n - 1:
                    assert not training.should_stop(i + 1, selector)
            assert n == 30 or training.should_stop(n, selector)
            assert selector == summary["selection"]
            checkpoint = torch.load(run_dir / "selected.pt", map_location="cpu", weights_only=False)
            last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            assert checkpoint["epoch"] == selector["best_epoch"] == metrics["classification"]["selected_epoch"]
            assert checkpoint["seed"] == last["seed"] == seed
            assert checkpoint["method"] == last["method"] == method
            assert last["epoch"] == n and last["selection"] == selector
            for group in checkpoint["optimizer"]["param_groups"]:
                assert group["lr"] == 0.001 and group["betas"] == (0.9, 0.999)
                assert group["eps"] == 1e-8 and group["weight_decay"] == 0
            for tensor in checkpoint["model"].values():
                assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
            restoration = next(r for r in audit["checkpoint_restorations"]
                               if r["seed"] == seed and r["method"] == method)
            assert restoration["model_sha256"] == model_hash(checkpoint["model"])
            orders = [o for o in audit["epoch_orders"] if o["seed"] == seed and o["method"] == method]
            assert len(orders) == n
            for t, order in enumerate(orders):
                assert order["zero_based_epoch"] == t
                assert order["permutation_sha256"] == array_hash(epoch_order(seed, t).numpy())
            with np.load(run_dir / "mnist_test_predictions.npz", allow_pickle=False) as saved:
                id_record = {k: saved[k].copy() for k in saved.files}
            with np.load(run_dir / "fashion_ood_predictions.npz", allow_pickle=False) as saved:
                ood_record = {k: saved[k].copy() for k in saved.files}
            for record in (id_record, ood_record):
                verify_finite(record)
                assert record["probabilities"].shape == (10000, 10)
                assert np.array_equal(record["index"], np.arange(10000))
                np.testing.assert_allclose(record["primary_uncertainty"], 1 - record["probabilities"].max(1), rtol=0, atol=0)
                np.testing.assert_allclose(np.exp(record["log_probabilities"]), record["probabilities"], rtol=1e-12, atol=1e-15)
            assert np.array_equal(id_record["labels"], datasets.test.targets.numpy())
            assert "labels" not in ood_record
            classification = classification_metrics(id_record["probabilities"], id_record["labels"], id_record["log_probabilities"])
            for key, value in classification.items():
                np.testing.assert_allclose(metrics["classification"][key], value, rtol=1e-12)
            primary = ood_metrics(id_record["primary_uncertainty"], ood_record["primary_uncertainty"])
            for key, value in (("primary_ood_auroc", primary["auroc"]), ("primary_ood_ap", primary["average_precision"])):
                np.testing.assert_allclose(metrics["ood_and_rejection"][key], value, rtol=1e-12)
            curve = risk_coverage(id_record["probabilities"], id_record["labels"], id_record["primary_uncertainty"], id_record["index"])
            for key, value in curve["metrics"].items():
                np.testing.assert_allclose(metrics["ood_and_rejection"][key], value, rtol=1e-12)
            with np.load(run_dir / "risk_coverage.npz", allow_pickle=False) as saved:
                for key in ("coverage", "risk", "order"):
                    assert np.array_equal(saved[key], curve[key])
            expected_plots = ["objective", "validation_accuracy", "validation_nll", "reliability",
                              "risk_coverage", "msp_histogram", "primary_uncertainty_histogram"]
            if method != "softmax":
                expected_plots.append("K_over_S_histogram")
                secondary = risk_coverage(id_record["probabilities"], id_record["labels"],
                                          id_record["evidential_uncertainty"], id_record["index"])
                secondary_rows.append({"method": method, "seed": seed, **secondary["metrics"]})
                np.savez_compressed(run_dir / "secondary_K_over_S_risk_coverage.npz",
                                    coverage=secondary["coverage"], risk=secondary["risk"], order=secondary["order"])
                write_json(run_dir / "secondary_K_over_S_risk_metrics.json", secondary["metrics"])
                fig, ax = plt.subplots()
                ax.plot(secondary["coverage"], secondary["risk"])
                ax.set(xlabel="Coverage", ylabel="Risk (error fraction)",
                       title=f"{method}, seed {seed}, secondary K/S uncertainty")
                save_plot(fig, run_dir / "plots" / "secondary_K_over_S_risk_coverage.png")
                expected_plots.append("secondary_K_over_S_risk_coverage")
                secondary_ood = ood_metrics(id_record["evidential_uncertainty"], ood_record["evidential_uncertainty"])
                np.testing.assert_allclose(metrics["ood_and_rejection"]["evidential_ood_auroc"], secondary_ood["auroc"], rtol=1e-12)
                np.testing.assert_allclose(metrics["ood_and_rejection"]["evidential_ood_ap"], secondary_ood["average_precision"], rtol=1e-12)
            for name in expected_plots:
                path = run_dir / "plots" / (name + ".png")
                assert path.exists() and path.stat().st_size > 1000, path
            # Explicit ID/OOD flags, original indices, probabilities and label semantics.
            records = {"original_index": np.concatenate([id_record["index"], ood_record["index"]]),
                       "ood_status": np.repeat([0, 1], [10000, 10000]),
                       "dataset_name": np.repeat(["MNIST", "FashionMNIST"], [10000, 10000]),
                       "dataset_label": np.concatenate([id_record["labels"], datasets.ood.targets.numpy()]),
                       "mnist_digit_label": np.concatenate([id_record["labels"], np.full(10000, -1)]),
                       "probabilities": np.concatenate([id_record["probabilities"], ood_record["probabilities"]]),
                       "log_probabilities": np.concatenate([id_record["log_probabilities"], ood_record["log_probabilities"]]),
                       "primary_uncertainty": np.concatenate([id_record["primary_uncertainty"], ood_record["primary_uncertainty"]])}
            if method != "softmax":
                records["evidential_uncertainty"] = np.concatenate([id_record["evidential_uncertainty"], ood_record["evidential_uncertainty"]])
            np.savez_compressed(run_dir / "per_example_records.npz", **records)
            run_statuses.append({"seed": seed, "method": method, "status": "PASS", "completed_epochs": n,
                                 "selected_epoch": checkpoint["epoch"],
                                 "training_seconds": summary["training_seconds"],
                                 "train_validation_seconds": summary["train_validation_seconds"]})
    write_table(directory / "secondary_K_over_S_risk_metrics.csv", secondary_rows)
    summary_table(directory / "secondary_K_over_S_risk_summary.csv", secondary_rows)
    write_table(directory / "run_statuses.csv", run_statuses)
    # Independently check every recorded aggregate uses arithmetic mean and n-1 SD.
    for per_seed_path, aggregate_path in (("classification_metrics.csv", "classification_summary.csv"),
                                          ("ood_rejection_metrics.csv", "ood_rejection_summary.csv"),
                                          ("secondary_K_over_S_risk_metrics.csv", "secondary_K_over_S_risk_summary.csv")):
        with (directory / per_seed_path).open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        with (directory / aggregate_path).open(encoding="utf-8") as f:
            aggregates = list(csv.DictReader(f))
        for aggregate in aggregates:
            group = [r for r in rows if r["method"] == aggregate["method"]]
            assert sorted(int(r["seed"]) for r in group) == [510, 511, 512]
            values = np.asarray([float(r[aggregate["metric"]]) for r in group])
            np.testing.assert_allclose(float(aggregate["mean"]), values.mean(), rtol=1e-12)
            np.testing.assert_allclose(float(aggregate["sample_sd"]), values.std(ddof=1), rtol=1e-12, atol=1e-15)
    completion = json.loads((directory / "completion.json").read_text())
    assert completion["runs"] == 9 and completion["status"] == "complete"
    assert fingerprint() == audit["source_fingerprint"]
    verification = {"status": "PASS", "protocol": "1.0", "runs": run_statuses,
                    "checks": {"all_nine_runs": True, "exact_seeds_and_methods": True,
                               "actual_initial_weights_paired_and_fresh": True, "actual_epoch_orders_paired": True,
                               "frozen_split_unchanged": True, "train_validation_only_selection": True,
                               "all_fits_before_test_ood_inference": True, "selected_weights_reloaded": True,
                               "one_id_and_one_ood_inference_per_run": True, "all_metrics_finite_and_verified": True,
                               "all_required_plots": True, "all_aggregate_means_and_sample_sd_verified": True,
                               "cpu_precision_and_optimizer": True, "core_source_and_protocol_unchanged": True},
                    "source_hashes": audit["core_source_hashes"], "execution_wall_seconds": audit["execution_wall_seconds"],
                    "bugs_or_corrections": [], "protocol_deviations": [],
                    "export_note": "Secondary K/S risk curves and explicit per-example metadata are generated solely from saved predictions, with no repeated training or inference. Fashion dataset_label is archival garment metadata; mnist_digit_label is -1 for OOD and no Fashion digit accuracy is computed."}
    write_json(directory / "final_verification.json", verification)
    (directory / "final_verification.txt").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    (directory / "artifact_notes.txt").write_text(verification["export_note"] + "\nPrimary risk_coverage.png uses 1-MSP. All histograms use 30 equal-width bins on [0,1].\n", encoding="utf-8")
    audit["status"] = "verified_complete"
    write_json(RESULTS_DIR / "final_execution_audit.json", audit)
    print("FINAL EXPERIMENT AND VERIFICATION COMPLETE: 9 runs; no tuning or extra inference.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Explicitly execute the already authorized nine final runs")
    if parser.parse_args().execute:
        execute()
    else:
        parser.print_help()
