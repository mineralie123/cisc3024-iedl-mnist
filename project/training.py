"""CPU training, validation-only selection, and restorable checkpoints."""
from pathlib import Path
import random
import time
import numpy as np
import torch
from config import P
from data import loader
from evaluation import collect_predictions, classification_metrics
from losses import objective, require_finite
from artifacts import write_json, plot_history


def configure_cpu():
    torch.set_num_threads(P.intra_threads)
    if torch.get_num_interop_threads() != P.inter_threads:
        try:
            torch.set_num_interop_threads(P.inter_threads)
        except RuntimeError as error:
            raise RuntimeError("Start a fresh Python process to set Protocol 1.0 inter-op threads") from error
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def optimizer_for(model):
    return torch.optim.Adam(model.parameters(), lr=P.lr, betas=P.betas,
                            eps=P.adam_eps, weight_decay=P.weight_decay)


def train_epoch(model, optimizer, method, batches, epoch):
    model.train()
    sums, examples = {}, 0
    for batch_index, (images, targets) in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        try:
            value, components = objective(model(images), targets, method, epoch)
            value.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    raise RuntimeError(f"Missing gradient: {name}")
                require_finite(f"gradient {name}", parameter.grad)
            optimizer.step()
            for name, parameter in model.named_parameters():
                require_finite(f"parameter {name}", parameter)
        except (FloatingPointError, RuntimeError) as error:
            raise RuntimeError(f"{method}, loss epoch={epoch}, batch={batch_index}: {error}") from error
        n = len(images)
        for name, tensor in {"objective": value, **components}.items():
            sums[name] = sums.get(name, 0.0) + tensor.detach().item() * n
        examples += n
    if examples == 0:
        raise ValueError("Empty training loader")
    return {name: total / examples for name, total in sums.items()}


def new_selection():
    return {"best_correct": -1, "best_nll": None, "best_epoch": None, "bad_epochs": 0}


def select_checkpoint(selection, validation, completed_epoch):
    """Mutates selection. Exact ties retain the earlier checkpoint."""
    better = (validation["correct"] > selection["best_correct"] or
              (validation["correct"] == selection["best_correct"]
               and validation["nll"] < selection["best_nll"]))
    if better:
        selection.update(best_correct=validation["correct"], best_nll=validation["nll"],
                         best_epoch=completed_epoch, bad_epochs=0)
    else:
        selection["bad_epochs"] += 1
    return better


def should_stop(completed_epochs, selection):
    # Patience counts all consecutive non-improvements, but the minimum is a gate.
    return completed_epochs >= P.minimum_epochs and selection["bad_epochs"] >= P.patience


def rng_states():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state()}


def save_checkpoint(path, model, optimizer, epoch, seed, method, selection, history):
    """epoch=completed epochs; next zero-based permutation index is epoch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "seed": seed, "method": method, "protocol": P.version,
                "rng_states": rng_states(),
                "data_order": {"next_epoch": epoch,
                               "next_generator_seed": seed + P.order_seed_offset + epoch,
                               "offset": P.order_seed_offset, "size": P.train_size,
                               "split_seed": P.split_seed},
                "selection": dict(selection), "history": history}, path)


def restore_checkpoint(path, model, optimizer=None, *, restore_rng=False):
    """Load only locally generated trusted checkpoints (pickle metadata included).

For final evaluation restore selected model weights only. For resuming training,
pass the optimizer and restore_rng=True to restore its state and all CPU RNGs.
"""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["protocol"] != P.version:
        raise ValueError("Checkpoint protocol mismatch")
    expected = state["seed"] + P.order_seed_offset + state["epoch"]
    if state["data_order"]["next_generator_seed"] != expected:
        raise ValueError("Checkpoint data-order metadata mismatch")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if restore_rng:
        random.setstate(state["rng_states"]["python"])
        np.random.set_state(state["rng_states"]["numpy"])
        torch.set_rng_state(state["rng_states"]["torch_cpu"])
    return state


def fit(model, method, seed, datasets, max_epochs, directory=None):
    """directory=None is the separate two-epoch pilot; it saves no weights."""
    seed_all(seed)
    optimizer = optimizer_for(model)
    history, selection = [], new_selection()
    directory = Path(directory) if directory is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
    start_run = time.perf_counter()
    for t in range(max_epochs):
        start_epoch = time.perf_counter()
        training = train_epoch(model, optimizer, method, loader(datasets.train, seed, t), t)
        training_seconds = time.perf_counter() - start_epoch
        validation_start = time.perf_counter()
        record, validation_loss = collect_predictions(
            model, method, loader(datasets.validation), epoch=t)
        validation = classification_metrics(record["probabilities"], record["labels"],
                                            record["log_probabilities"])
        validation_seconds = time.perf_counter() - validation_start
        epoch_seconds = time.perf_counter() - start_epoch
        completed = t + 1
        improved = select_checkpoint(selection, validation, completed)
        row = {"completed_epochs": completed, "loss_epoch": t,
               "kl_annealing": min(1.0, t / P.kl_anneal_epochs),
               "train": training, "validation": validation, "validation_loss": validation_loss,
               "training_seconds": training_seconds, "validation_seconds": validation_seconds,
               "epoch_seconds": epoch_seconds, "selected": improved}
        history.append(row)
        if directory is not None:
            if improved:
                save_checkpoint(directory / "selected.pt", model, optimizer, completed,
                                seed, method, selection, history)
            save_checkpoint(directory / "last.pt", model, optimizer, completed,
                            seed, method, selection, history)
            write_json(directory / "history.json", history)
        print(f"{method} seed={seed} completed_epoch={completed} wall={epoch_seconds:.2f}s", flush=True)
        if should_stop(completed, selection):
            break
    run_seconds = time.perf_counter() - start_run
    if directory is not None:
        plot_history(history, directory / "plots", f"{method}, seed {seed}")
    return {"history": history, "selection": selection,
            "training_seconds": sum(r["training_seconds"] for r in history),
            "train_validation_seconds": sum(r["epoch_seconds"] for r in history),
            "run_seconds": run_seconds, "completed_epochs": len(history)}
