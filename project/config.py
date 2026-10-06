"""Protocol 1.0 constants. No command-line hyperparameter overrides."""
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets"
RESULTS_DIR = ROOT / "results"
SPLIT_PATH = DATA_DIR / "mnist_split_3024.npz"
VERIFICATION_DIR = ROOT / "evidence" / "verification"
METHODS = ("softmax", "edl", "iedl")


@dataclass(frozen=True)
class Protocol:
    version: str = "1.0"
    split_seed: int = 3024
    seeds: tuple = (510, 511, 512)
    pilot_seed: int = 999
    classes: int = 10
    train_size: int = 10000
    validation_size: int = 2000
    train_per_class: int = 1000
    validation_per_class: int = 200
    mnist_test_size: int = 10000
    fashion_test_size: int = 10000
    fashion_train: bool = False
    kmnist: bool = False
    preprocessing: str = "ToTensor only; no normalization, resizing or augmentation"
    train_batch: int = 64
    eval_batch: int = 256
    workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    intra_threads: int = 4
    inter_threads: int = 1
    device: str = "cpu"
    model_dtype: str = "float32"
    loss_dtype: str = "float64"
    architecture: str = "conv1-16-32-32(k5,p2,pool2);288-32-32-32-10;ReLU"
    optimizer: str = "Adam"
    lr: float = 0.001
    betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    scheduler: bool = False
    gradient_clipping: bool = False
    fisher_coefficient: float = 0.005
    kl_anneal_epochs: int = 10
    loss_epoch_indexing: str = "zero-based t; KL=min(1,t/10); completed epoch=t+1"
    order_seed_offset: int = 100000
    minimum_epochs: int = 15
    patience: int = 5
    validation_frequency: int = 1
    selection: str = "max correct count, min predictive NLL, earliest epoch"
    pilot_epochs: int = 2
    budget_thresholds: tuple = ((160.0, 30), (240.0, 20), (320.0, 15))
    ece_bins: int = 15
    histogram_bins: int = 30
    histogram_range: tuple = (0.0, 1.0)
    histogram_note: str = "30 equal-width bins over [0,1], explicitly confirmed by user before results"
    primary_uncertainty: str = "1-max(q)"
    secondary_uncertainty: str = "K/S (EDL and I-EDL only)"
    mutual_information: bool = False
    ood_positive: int = 1
    coverage_levels: tuple = (0.5, 0.8, 0.9, 1.0)
    aurc_definition: str = "mean risk over all retained counts 1..N"
    ece_definition: str = "equal-width [lower,upper) bins; final bin includes 1"
    sample_std_ddof: int = 1


P = Protocol()


def snapshot():
    return {"protocol": asdict(P), "paths": {"root": str(ROOT),
            "data": str(DATA_DIR), "split": str(SPLIT_PATH)}, "methods": METHODS}


def fingerprint():
    """Tie pilot and final runs to both settings and exact source files."""
    h = sha256(json.dumps(snapshot(), sort_keys=True).encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def timing_budget(slowest_seconds):
    import math
    if not math.isfinite(slowest_seconds) or slowest_seconds <= 0:
        raise ValueError("Pilot duration must be positive and finite")
    for threshold, epochs in P.budget_thresholds:
        if slowest_seconds <= threshold:
            return epochs
    return None  # Stop: Protocol 1.0 is impractical on the measured hardware.
