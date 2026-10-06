"""Bounded verification: one MNIST training batch per method; no pilot/test fit."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import random
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import polygamma, digamma, gammaln
from config import P, METHODS, ROOT, DATA_DIR, VERIFICATION_DIR, RESULTS_DIR, timing_budget
from models import ReducedCNN, paired_models
from losses import (objective, evidence_parameters, fisher_terms, masked_kl,
                    prediction, require_finite)
from data import load_datasets, loader, epoch_order, make_split
from training import (configure_cpu, seed_all, optimizer_for, train_epoch, new_selection,
                      select_checkpoint, should_stop, save_checkpoint, restore_checkpoint)
from evaluation import (classification_metrics, calibration, risk_coverage, ood_metrics,
                        collect_predictions)
from artifacts import (write_json, write_snapshot, plot_history, plot_predictions,
                       summary_table, environment)


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        assert a == b


def synthetic_record():
    q = np.zeros((4, P.classes), dtype=np.float64)
    q[:, :2] = [[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.9, 0.1]]
    labels = np.array([0, 1, 0, 1])
    with np.errstate(divide="ignore"):
        logq = np.log(q)
    return {"probabilities": q, "log_probabilities": logq, "labels": labels,
            "index": np.arange(4), "primary_uncertainty": 1 - q.max(1),
            "evidential_uncertainty": np.array([0.2, 0.3, 0.4, 0.1])}


def check_loss_mathematics():
    # Independent NumPy/SciPy calculations, not a restatement using loss helpers.
    z = np.linspace(-3, 4, 30).reshape(3, 10)
    target = np.array([0, 4, 9])
    alpha = np.logaddexp(0, z) + 1
    S = alpha.sum(1, keepdims=True)
    q = alpha / S
    one_hot = np.eye(10)[target]
    var = alpha * (S - alpha) / (S**2 * (S + 1))
    masked = alpha * (1 - one_hot) + one_hot
    masked_sum = masked.sum(1)
    kl = (gammaln(masked_sum) - gammaln(10) - gammaln(masked).sum(1)
          + ((masked - 1) * (digamma(masked) - digamma(masked_sum)[:, None])).sum(1))
    weights = polygamma(1, alpha)
    matrices = np.array([np.diag(w) - polygamma(1, s.item()) for w, s in zip(weights, S)])
    sign, logdet = np.linalg.slogdet(matrices)
    assert np.all(sign == 1)
    logits = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor(target)
    for epoch in (0, 1, 9, 10, 20):
        scale = min(1, epoch / 10)
        edl, e = objective(logits, labels, "edl", epoch)
        iedl, i = objective(logits, labels, "iedl", epoch)
        expected_edl = (((one_hot - q)**2).sum(1) + var.sum(1) + scale * kl).mean()
        expected_iedl = ((((one_hot - q)**2 + var) * weights).sum(1)
                         - P.fisher_coefficient * logdet + scale * kl).mean()
        np.testing.assert_allclose(edl.detach().numpy(), expected_edl, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(iedl.detach().numpy(), expected_iedl, rtol=1e-9, atol=1e-10)
        assert "negative_fisher_logdet" not in e
        assert_same(e["masked_kl"], i["masked_kl"])
    ce, c = objective(logits, labels, "softmax", 1)
    torch.testing.assert_close(ce, F.cross_entropy(logits, labels))
    assert set(c) == {"cross_entropy"}
    _, a, _, _ = evidence_parameters(logits)
    trigamma, fisher, negative_logdet = fisher_terms(a)
    torch.testing.assert_close(negative_logdet, -torch.linalg.slogdet(fisher).logabsdet)
    torch.testing.assert_close(negative_logdet,
                              -(trigamma.log().sum(-1) + torch.log(
                                  1 - (torch.polygamma(1, a.sum(-1))[:, None] / trigamma).sum(-1))))
    for value in (1.0, 100.0, 100000.0):
        fisher_terms(torch.full((2, 10), value, dtype=torch.float64))
    uniform_kl = masked_kl(torch.ones_like(a), F.one_hot(labels, 10).double())
    torch.testing.assert_close(uniform_kl, torch.zeros_like(uniform_kl), atol=1e-12, rtol=0)
    grad = torch.autograd.grad(trigamma.sum(), a, retain_graph=True)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    value, components = objective(logits, labels, "iedl", 3)
    for name in ("fisher_mse", "fisher_variance", "negative_fisher_logdet"):
        grad = torch.autograd.grad(components[name], logits, retain_graph=True)[0]
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    # Finite differences check the total differentiable Fisher objective.
    derivative = torch.autograd.grad(value, logits)[0][1, 4].item()
    perturbation = 1e-5
    plus, minus = logits.detach().clone(), logits.detach().clone()
    plus[1, 4] += perturbation
    minus[1, 4] -= perturbation
    difference = (objective(plus, labels, "iedl", 3)[0].item()
                  - objective(minus, labels, "iedl", 3)[0].item()) / (2 * perturbation)
    np.testing.assert_allclose(derivative, difference, rtol=1e-5, atol=1e-8)
    try:
        require_finite("intentional smoke NaN", torch.tensor(float("nan")))
    except FloatingPointError:
        pass
    else:
        raise AssertionError("Nonfinite values must fail")
    try:
        fisher_terms(torch.full((1, 10), -0.5, dtype=torch.float64))
    except FloatingPointError:
        pass
    else:
        raise AssertionError("Invalid Fisher matrix must fail")
    return "PASS: independent SciPy objectives, KL schedule, logdet identities, Fisher gradients, finite differences and failure diagnostics"


def check_metrics():
    record = synthetic_record()
    q, labels = record["probabilities"], record["labels"]
    metrics = classification_metrics(q, labels, record["log_probabilities"])
    assert metrics["accuracy"] == 0.5 and metrics["correct"] == 2
    np.testing.assert_allclose(metrics["nll"], -np.log([0.8, 0.7, 0.4, 0.1]).mean())
    np.testing.assert_allclose(metrics["brier"], np.mean([0.08, 0.18, 0.72, 1.62]))
    np.testing.assert_allclose(metrics["ece"], 0.5)  # Four distinct confidence bins.
    curve = risk_coverage(q, labels, record["primary_uncertainty"])
    np.testing.assert_allclose(curve["risk"], [1, 0.5, 1/3, 0.5])
    np.testing.assert_allclose(curve["metrics"]["aurc"], 7/12)
    assert curve["metrics"]["risk_50"] == 0.5
    assert all(curve["metrics"][f"risk_{c}"] == 0.5 for c in (80, 90, 100))
    tied = risk_coverage(q, labels, np.ones(4), np.array([3, 2, 1, 0]))
    assert tied["order"].tolist() == [3, 2, 1, 0]
    perfect = np.eye(10)[[0, 1]]
    assert calibration(perfect, [0, 1])["ece"] == 0
    assert calibration(perfect, [0, 1])["count"][-1] == 2
    scores = ood_metrics([0.1, 0.2], [0.8, 0.9])
    assert scores == {"auroc": 1.0, "average_precision": 1.0}
    tied_ood = ood_metrics([0.5, 0.5], [0.5, 0.5])
    assert tied_ood == {"auroc": 0.5, "average_precision": 0.5}
    return "PASS: accuracy, NLL, class-summed Brier, ECE including confidence=1, AURC, coverage risks, index tie breaks, OOD-positive AUROC/AP"


def run_smoke(download=False):
    configure_cpu()
    seed_all(P.pilot_seed)
    VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    directory = RESULTS_DIR / "smoke"
    directory.mkdir(parents=True, exist_ok=True)
    report = {"status": "in_progress", "scope": "verification only; not experimental results",
              "pilot_ran": False, "full_training_ran": False, "final_test_evaluation_ran": False,
              "dataset_downloads_allowed": download, "environment": environment(), "checks": {}}
    checks = report["checks"]
    try:
        sources = sorted(Path(__file__).parent.glob("*.py"))
        for path in sources:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        report["created_source_files"] = [str(path.relative_to(ROOT)) for path in sources]
        report["readme"] = "project/README.md"
        report["implementation_notes"] = "evidence/verification/implementation_notes.txt"
        report["protocol_summary_sha256"] = sha256((ROOT / "evidence" / "summaries" /
                                                    "04_frozen_protocol_summary.txt").read_bytes()).hexdigest()
        checks["syntax_imports"] = "PASS: all modules imported and source files parsed"
        write_snapshot(directory)
        before = {str(p.relative_to(DATA_DIR)) for p in DATA_DIR.rglob("*") if p.is_file()} if DATA_DIR.exists() else set()
        datasets = load_datasets(download=download)
        after = {str(p.relative_to(DATA_DIR)) for p in DATA_DIR.rglob("*") if p.is_file()}
        report["new_dataset_files"] = sorted(after - before)
        train_y = datasets.train.dataset.targets[datasets.train_indices].numpy()
        val_y = datasets.validation.dataset.targets[datasets.validation_indices].numpy()
        assert len(datasets.train) == P.train_size and len(datasets.validation) == P.validation_size
        assert np.bincount(train_y, minlength=10).tolist() == [1000] * 10
        assert np.bincount(val_y, minlength=10).tolist() == [200] * 10
        assert len(np.unique(datasets.train_indices)) == P.train_size
        assert len(np.unique(datasets.validation_indices)) == P.validation_size
        assert len(np.intersect1d(datasets.train_indices, datasets.validation_indices)) == 0
        assert len(datasets.test) == len(datasets.ood) == 10000
        assert_same(make_split(datasets.train.dataset.targets),
                    (datasets.train_indices, datasets.validation_indices))
        assert not list((DATA_DIR / "FashionMNIST").rglob("train-*"))
        checks["split"] = {"status": "PASS", "train": 10000, "validation": 2000,
                           "mnist_test": 10000, "fashion_test": 10000,
                           "train_class_counts": np.bincount(train_y).tolist(),
                           "validation_class_counts": np.bincount(val_y).tolist(),
                           "disjoint": True, "unused_training_examples": 48000,
                           "fashion_train_downloaded": False}
        for dataset in (datasets.train, datasets.validation, datasets.test, datasets.ood):
            x, _ = dataset[0]
            assert x.shape == (1, 28, 28) and x.dtype == torch.float32
            assert 0 <= x.min() <= x.max() <= 1
        train_loader = loader(datasets.train, P.pilot_seed, 0)
        assert train_loader.num_workers == 0 and not train_loader.pin_memory and not train_loader.drop_last
        assert len(train_loader) == 157
        assert len(list(train_loader.batch_sampler)[-1]) == 16
        for dataset in (datasets.validation, datasets.test, datasets.ood):
            batches = loader(dataset)
            assert len(list(batches.batch_sampler)[-1]) == len(dataset) % P.eval_batch
        checks["loaders"] = "PASS: ToTensor-only shapes/range; CPU; workers=0; no pinning; partial batches preserved"
        paired = paired_models(P.pilot_seed)
        for seed in (*P.seeds, P.pilot_seed):
            models = paired_models(seed)
            for method in METHODS:
                assert_same(models["softmax"].state_dict(), models[method].state_dict())
        checks["paired_initialization"] = "PASS: exact state equality for every final seed and pilot seed"
        for seed in (*P.seeds, P.pilot_seed):
            for t in (0, 1, 14, 29):
                expected = torch.randperm(P.train_size, generator=torch.Generator().manual_seed(seed + 100000 + t))
                orders = [torch.tensor(list(loader(datasets.train, seed, t).sampler)) for _ in METHODS]
                for order in orders:
                    assert_same(order, expected)
                assert torch.unique(orders[0]).numel() == P.train_size
        checks["paired_order"] = "PASS: s+100000+t CPU generator; full identical permutations for all methods"
        model = paired["softmax"]
        x, y = next(iter(train_loader))
        assert model.features(x[:2]).shape == (2, 32, 3, 3)
        assert model(x[:2]).shape == (2, 10)
        assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters())
        assert not any(isinstance(m, (torch.nn.Dropout, torch.nn.modules.batchnorm._BatchNorm))
                       for m in model.modules())
        count = sum(p.numel() for p in model.parameters())
        assert count == 50570
        report["parameter_count"] = count
        checks["architecture"] = "PASS: (N,1,28,28)->(N,32,3,3)->(N,10); 50,570 float32 CPU parameters"
        checks["loss_mathematics"] = check_loss_mathematics()
        component_names = {}
        for method in METHODS:
            opt = optimizer_for(paired[method])
            logits = paired[method](x[:4])
            value, components = objective(logits, y[:4], method, 0)
            assert value.dtype == torch.float64
            q, _, u = prediction(logits, method)
            assert q.dtype == torch.float64
            assert u is None if method == "softmax" else u is not None
            component_names[method] = list(components)
            train_epoch(paired[method], opt, method, [(x, y)], 0)  # ONE real training batch.
            if method == "softmax":
                saved_optimizer = opt
        report["real_training_batches_per_method"] = 1
        checks["loss_components"] = component_names
        checks["training_gradients"] = "PASS: one CPU batch per method, finite losses/gradients/parameters; independent Adam states"
        checks["metrics"] = check_metrics()
        # Check weighted evaluation with unequal batch sizes using synthetic inputs.
        synthetic_x = torch.rand(5, 1, 28, 28)
        synthetic_y = torch.tensor([0, 1, 2, 3, 4])
        batches = [(synthetic_x[:3], synthetic_y[:3]), (synthetic_x[3:], synthetic_y[3:])]
        for method in METHODS:
            _, aggregate = collect_predictions(paired[method], method, batches, epoch=2)
            parts = [objective(paired[method](bx), by, method, 2)[0].item() * len(bx) for bx, by in batches]
            np.testing.assert_allclose(aggregate["objective"], sum(parts) / 5, rtol=1e-12)
            record, _ = collect_predictions(paired[method], method, batches, labelled=False)
            assert "labels" not in record
        checks["weighted_validation_and_ood_labels"] = "PASS: 3+2 synthetic batch weighting; OOD records exclude labels"
        selection = new_selection()
        assert select_checkpoint(selection, {"correct": 5, "nll": 2.0}, 1)
        assert select_checkpoint(selection, {"correct": 6, "nll": 3.0}, 2)
        assert select_checkpoint(selection, {"correct": 6, "nll": 2.5}, 3)
        assert not select_checkpoint(selection, {"correct": 6, "nll": 2.5}, 4)
        assert not select_checkpoint(selection, {"correct": 5, "nll": 1.0}, 5)
        assert selection["best_epoch"] == 3
        selection["bad_epochs"] = 5
        assert not should_stop(14, selection) and should_stop(15, selection)
        checks["selection_and_stopping"] = "PASS: correct-count, NLL, earlier-epoch precedence; min 15; patience 5"
        checkpoints = directory / "smoke_checkpoint.pt"
        seed_all(P.pilot_seed)
        save_checkpoint(checkpoints, paired["softmax"], saved_optimizer, 1,
                        P.pilot_seed, "softmax", selection, [])
        expected_draws = (random.random(), np.random.rand(), torch.rand(2))
        restored = ReducedCNN()
        restored_optimizer = optimizer_for(restored)
        state = restore_checkpoint(checkpoints, restored, restored_optimizer, restore_rng=True)
        actual_draws = (random.random(), np.random.rand(), torch.rand(2))
        assert_same(expected_draws, actual_draws)
        assert_same(restored.state_dict(), paired["softmax"].state_dict())
        assert_same(restored_optimizer.state_dict(), saved_optimizer.state_dict())
        assert_same(state["selection"], selection)
        assert state["data_order"]["next_generator_seed"] == 101000
        assert_same(epoch_order(state["seed"], state["data_order"]["next_epoch"]), epoch_order(999, 1))
        assert_same(restored(x[:2]), paired["softmax"](x[:2]))
        checks["checkpoint_restore"] = "PASS: exact model, optimizer, RNG draws, selector, epoch, next permutation and predictions"
        for duration, budget in ((0.01, 30), (160, 30), (160.001, 20), (240, 20),
                                 (240.001, 15), (320, 15), (320.001, None)):
            assert timing_budget(duration) == budget
        for invalid in (0, -1, float("nan"), float("inf")):
            try:
                timing_budget(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid timing must fail")
        checks["timing_budget"] = "PASS: all thresholds and impractical/invalid-runtime branches; pilot NOT executed"
        synthetic = synthetic_record()
        ood = {k: v.copy() for k, v in synthetic.items() if k != "labels"}
        history = [{"completed_epochs": t + 1, "train": {"objective": 1/(t+1)},
                    "validation_loss": {"objective": 1.1/(t+1)},
                    "validation": {"accuracy": 0.5, "nll": 1.0}} for t in range(2)]
        plot_history(history, directory / "synthetic_plots", "SYNTHETIC SMOKE DATA")
        plot_predictions(synthetic, ood, directory / "synthetic_plots", "SYNTHETIC SMOKE DATA")
        plots = sorted((directory / "synthetic_plots").glob("*.png"))
        assert len(plots) == 8 and all(p.stat().st_size > 1000 for p in plots)
        rows = [{"method": method, "seed": seed, "synthetic_metric": i}
                for method in METHODS for i, seed in enumerate(P.seeds)]
        summary_table(directory / "synthetic_summary.csv", rows)
        summary = (directory / "synthetic_summary.csv").read_text(encoding="utf-8")
        assert "1.0,1.0" in summary
        checks["plots_and_summary"] = "PASS: 8 synthetic PNGs and ddof=1 three-seed summary; no test results"
        report["synthetic_plot_files"] = [str(p.relative_to(ROOT)) for p in plots]
        checks["cpu_protocol"] = "PASS: 4 intra-op, 1 inter-op, deterministic algorithms enabled, no CUDA execution"
        assert torch.get_num_threads() == 4 and torch.get_num_interop_threads() == 1
        assert torch.are_deterministic_algorithms_enabled()
        report["status"] = "PASS"
        report["deviations"] = []
        report["clarifications"] = ["30 histogram bins confirmed by user before results",
                                    "Loss epoch zero-based (official convention); completed epochs one-based",
                                    "Patience counts non-improvements from start, with a 15-epoch stopping gate"]
    except Exception:
        report["status"] = "FAIL"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                                   for p in Path(__file__).parent.glob("*.py")}
        write_json(VERIFICATION_DIR / "verification.json", report)
        attempt = 1
        while (VERIFICATION_DIR / f"attempt_{attempt:02d}.json").exists():
            attempt += 1
        write_json(VERIFICATION_DIR / f"attempt_{attempt:02d}.json", report)
        lines = ["CISC3024 Protocol 1.0 implementation verification", "Status: " + report["status"],
                 "Scope: bounded smoke checks; no pilot or full experiment", ""]
        for name, value in checks.items():
            lines.append(name + ": " + (value if isinstance(value, str) else json.dumps(value)))
        lines += ["", "Details:", json.dumps(report, indent=2)]
        (VERIFICATION_DIR / "verification.txt").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "parameter_count": report["parameter_count"],
                      "real_training_batches_per_method": 1, "pilot_ran": False,
                      "final_test_evaluation_ran": False}, indent=2), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    run_smoke(download=parser.parse_args().download)
