"""
Self-Pruning Neural Network on CIFAR-10
=======================================

Tredence Analytics — AI Engineering Intern case study.

What this script does, in plain English
---------------------------------------
We train an image classifier on CIFAR-10 (10 categories of tiny 32x32 photos
like "cat", "ship", "frog"). The twist: the network is allowed to **switch off
its own connections during training**. Each connection (each weight in a linear
layer) is paired with a learnable "gate" -- a number between 0 and 1. The
effective weight used in the forward pass is `weight * gate`. If a gate slides
down to 0, that connection is effectively dead.

We add a penalty to the loss that grows with the sum of all gates. The
optimizer then has two competing pressures:
  1. classification loss -> "keep gates open so I can predict accurately"
  2. sparsity penalty    -> "close as many gates as possible"
A hyperparameter `lambda` (lam) controls how strong the sparsity pressure is.

Run it
------
    # one configuration:
    python self_pruning.py --lam 5e-6 --epochs 25

    # all four configurations from the report (baseline + 3 lambdas):
    python self_pruning.py --run-all

    # quick smoke test (1 epoch on a tiny subset, runs on CPU):
    python self_pruning.py --lam 1e-6 --epochs 1 --subset 1024

Running in a Jupyter / Google Colab notebook
--------------------------------------------
Just paste the whole file into a cell and run it. The script detects the
notebook environment and skips CLI-arg parsing (which would otherwise choke
on Jupyter's own kernel arguments). It defaults to the full sweep
(--run-all). To run a different configuration from a notebook, call:

    main(["--lam", "5e-6", "--epochs", "25"])
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")  # write plot to a file without opening a GUI window
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 0) -> None:
    """Make every run produce the same numbers, given the same hardware.

    PyTorch, NumPy and Python's `random` all have their own RNGs, so we
    seed all three. cudnn.deterministic forces some GPU kernels to pick
    deterministic algorithms (slightly slower, but reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 1. PrunableLinear -- the core piece graded by the case study
# ---------------------------------------------------------------------------

class PrunableLinear(nn.Module):
    """A drop-in replacement for nn.Linear with a learnable gate per weight.

    Every weight in this layer is paired with a `gate_score` (raw real number).
    During the forward pass we squash gate_scores through a sigmoid to get
    `gates` in [0, 1], then multiply element-wise with the weight matrix:

        effective_weight = weight * sigmoid(gate_score)

    If gate -> 0, the corresponding connection is "pruned" (its contribution
    to the output is zero).

    Crucially, `gate_scores` is registered as an `nn.Parameter`, so the
    optimizer updates it just like the weights themselves -- the network
    *learns* which connections to keep.
    """

    def __init__(self, in_features: int, out_features: int, gate_init: float = 2.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard linear-layer parameters.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        # The new piece: one gate score per weight, same shape as `weight`.
        # We initialise scores at +2.0 so sigmoid(2.0) ~ 0.88. That means the
        # network *starts* close to a normal linear layer (gates almost fully
        # open), and the L1 pressure drives unimportant gates down from there.
        # Initialising at 0 (gates = 0.5) would halve the signal at init and
        # slow convergence.
        self.gate_scores = nn.Parameter(torch.full_like(self.weight, gate_init))

        # Same init scheme as torch.nn.Linear so the layer behaves identically
        # at the start (before pruning kicks in).
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def gates(self) -> torch.Tensor:
        """Return the current gate values in [0, 1]."""
        return torch.sigmoid(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Element-wise multiply: gradient flows naturally to BOTH `weight`
        # (multiplied by `gates`) and `gate_scores` (through sigmoid).
        # F.linear(x, W, b) computes x @ W.T + b -- a standard linear layer.
        effective_weight = self.weight * self.gates()
        return F.linear(x, effective_weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, gated=True"


# ---------------------------------------------------------------------------
# 2. The network -- a plain MLP made entirely of PrunableLinear layers
# ---------------------------------------------------------------------------

class PrunableMLP(nn.Module):
    """4-layer feed-forward network for CIFAR-10, every linear is prunable.

    Input: a 32x32x3 image flattened to a 3072-vector.
    Output: 10 class logits (airplane, car, bird, ..., truck).
    """

    def __init__(self, hidden_dims=(1024, 512, 256), num_classes: int = 10):
        super().__init__()
        in_dim = 3 * 32 * 32  # 3072 -- CIFAR-10 image flattened
        layers = []
        for h in hidden_dims:
            layers.append(PrunableLinear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        layers.append(PrunableLinear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (batch, 3, 32, 32); flatten everything except batch dim.
        x = x.flatten(start_dim=1)
        return self.net(x)


# ---------------------------------------------------------------------------
# 3. Sparsity loss -- the L1 penalty on gates
# ---------------------------------------------------------------------------

def sparsity_loss(model: nn.Module) -> torch.Tensor:
    """Sum of every gate value in the network.

    Why L1 (sum of absolute values) encourages sparsity:
      - sigmoid outputs are always in [0, 1], so the absolute value is the
        value itself; L1 norm = simple sum.
      - The gradient of `sigmoid(s)` with respect to `s` is sigmoid(s)*(1-sigmoid(s)),
        which is positive everywhere. So the optimizer always wants to push
        scores DOWN to reduce the penalty. The classification loss only
        protects scores whose corresponding weight is actually useful;
        all others get driven into the saturated left tail of the sigmoid
        where the gate is effectively 0.
      - Compared to L2, L1 has a constant gradient near zero, so it actually
        *reaches* zero rather than getting smaller and smaller.

    This function uses `sum` of gates (not `mean`), exactly as the brief
    specifies. That means lambda needs to be small (the sum is in the millions
    for our ~3.8M-weight model).
    """
    total = torch.zeros((), device=next(model.parameters()).device)
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            total = total + module.gates().sum()
    return total


# ---------------------------------------------------------------------------
# 4. Sparsity & accuracy evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_sparsity(model: nn.Module, threshold: float = 1e-2) -> dict:
    """Fraction of gates whose value is below `threshold` (effectively zero).

    Returns overall sparsity and per-layer sparsity. The brief uses 1e-2
    as the threshold for "effectively pruned".
    """
    per_layer = {}
    total_gates = 0
    total_pruned = 0
    for name, module in model.named_modules():
        if isinstance(module, PrunableLinear):
            gates = module.gates()
            n = gates.numel()
            pruned = (gates < threshold).sum().item()
            per_layer[name] = {
                "n_weights": n,
                "n_pruned": pruned,
                "sparsity_pct": 100.0 * pruned / n,
            }
            total_gates += n
            total_pruned += pruned
    return {
        "overall_sparsity_pct": 100.0 * total_pruned / total_gates,
        "total_gates": total_gates,
        "total_pruned": total_pruned,
        "per_layer": per_layer,
    }


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Standard test-set accuracy as a percentage."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# 5. Data loaders
# ---------------------------------------------------------------------------

# CIFAR-10 channel-wise mean and std (commonly used values, computed over the
# training set). Normalising to roughly N(0, 1) helps the optimizer.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_loaders(batch_size: int = 128, data_dir: str = "./data", subset: int | None = None):
    """Download CIFAR-10 (if needed) and return train/test DataLoaders.

    `subset` takes the first N training samples; useful for smoke-testing.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # PIL image -> Tensor in [0, 1]
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    if subset is not None:
        train_set = Subset(train_set, range(min(subset, len(train_set))))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    lam: float
    test_accuracy: float
    sparsity_pct: float
    per_layer_sparsity: dict
    epochs: int
    seconds: float


def train_one_config(
    lam: float,
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 1e-3,
    subset: int | None = None,
    device: torch.device | None = None,
    seed: int = 0,
) -> tuple[PrunableMLP, RunResult]:
    """Train one model with the given lambda, return (trained model, results)."""
    set_seed(seed)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    model = PrunableMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader, test_loader = get_loaders(batch_size=batch_size, subset=subset)

    print(f"\n=== lambda = {lam:g} | device = {device} | epochs = {epochs} ===")
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_ce = 0.0
        epoch_sparsity = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            ce = F.cross_entropy(logits, y)
            sp = sparsity_loss(model)  # sum of all gates
            loss = ce + lam * sp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_ce += ce.item()
            epoch_sparsity += sp.item()
            n_batches += 1

        # End-of-epoch report so you can watch the gates shut.
        sparsity_info = compute_sparsity(model)
        test_acc = evaluate_accuracy(model, test_loader, device)
        print(
            f"epoch {epoch:2d}/{epochs} | "
            f"ce={epoch_ce / n_batches:.4f} | "
            f"sum_gates={epoch_sparsity / n_batches:,.0f} | "
            f"test_acc={test_acc:.2f}% | "
            f"sparsity={sparsity_info['overall_sparsity_pct']:.1f}%"
        )

    elapsed = time.time() - start
    final_sparsity = compute_sparsity(model)
    final_acc = evaluate_accuracy(model, test_loader, device)
    result = RunResult(
        lam=lam,
        test_accuracy=final_acc,
        sparsity_pct=final_sparsity["overall_sparsity_pct"],
        per_layer_sparsity=final_sparsity["per_layer"],
        epochs=epochs,
        seconds=elapsed,
    )
    print(f"--> finished lambda={lam:g}: acc={final_acc:.2f}%  sparsity={result.sparsity_pct:.1f}%  "
          f"({elapsed:.1f}s)")
    return model, result


# ---------------------------------------------------------------------------
# 7. Plotting
# ---------------------------------------------------------------------------

def plot_gate_distribution(model: nn.Module, out_path: Path, title: str = "Gate value distribution") -> None:
    """Histogram of every gate value in the trained model.

    A successful pruning run shows two clusters: a tall spike near 0 (pruned
    connections) and a smaller cluster near 1 (the connections the network
    decided to keep).
    """
    all_gates = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                all_gates.append(module.gates().detach().cpu().flatten())
    gates = torch.cat(all_gates).numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gates, bins=50, range=(0, 1), color="steelblue", edgecolor="black")
    ax.set_yscale("log")  # log-scale so the small cluster near 1 is visible
    ax.set_xlabel("gate value (sigmoid of gate_score)")
    ax.set_ylabel("count (log scale)")
    ax.set_title(title)
    ax.axvline(1e-2, color="red", linestyle="--", linewidth=1, label="pruned threshold (1e-2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved plot to {out_path}")


# ---------------------------------------------------------------------------
# 8. Gradient-flow self-check (catches the most common bug)
# ---------------------------------------------------------------------------

def gradient_flow_check(device: torch.device) -> None:
    """Confirm that gradients reach `gate_scores` (not just `weight`/`bias`).

    The brief explicitly calls out gradient flow as the challenge of this
    layer, so we assert it here. Run once at startup; it costs ~10ms.
    """
    print("running gradient-flow self-check...")
    model = PrunableMLP().to(device)
    x = torch.randn(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    loss = F.cross_entropy(model(x), y) + 1e-6 * sparsity_loss(model)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert p.grad.abs().sum().item() > 0, f"zero gradient for {name}"
    print("  ok: weights, biases AND gate_scores all received nonzero gradients.\n")


# ---------------------------------------------------------------------------
# 9. CLI / notebook entry point
# ---------------------------------------------------------------------------

def _running_in_notebook() -> bool:
    """True if this code is executing inside a Jupyter or Google Colab cell.

    We use this to skip argparse when the script is pasted into a notebook
    (Jupyter passes its own `-f /path/to/kernel.json` argument to the kernel,
    which argparse would reject as 'unrecognized arguments').
    """
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lam", type=float, default=5e-6, help="sparsity weight lambda (default: 5e-6)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subset", type=int, default=None,
                        help="use only the first N training samples (for smoke tests)")
    parser.add_argument("--run-all", action="store_true",
                        help="train baseline + low + medium + high lambdas and write results.json")
    parser.add_argument("--out-dir", type=str, default="outputs")

    # Notebook fallback: when no explicit argv is given AND we're running
    # inside a Jupyter/Colab kernel, ignore sys.argv (which contains the
    # kernel-launcher args) and default to the full sweep.
    if argv is None and _running_in_notebook():
        print("[notebook detected] skipping CLI args, running --run-all with defaults.")
        print("[notebook detected] to customise, call e.g. main(['--lam', '5e-6', '--epochs', '25']).\n")
        argv = ["--run-all"]

    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    gradient_flow_check(device)

    if args.run_all:
        # Lambdas are calibrated to our ~3.8M-weight model: at init, the
        # sparsity sum is ~3.3M and CE is ~2.3, so lam=5e-6 puts the two
        # terms on the same order of magnitude.
        configs = [
            ("baseline", 0.0),
            ("low",      5e-7),
            ("medium",   5e-6),
            ("high",     5e-5),
        ]
        results = []
        best = None  # (accuracy * (1 + sparsity_fraction), model, label)
        for label, lam in configs:
            model, r = train_one_config(
                lam=lam, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, subset=args.subset, device=device, seed=args.seed,
            )
            record = {"label": label, **asdict(r)}
            results.append(record)
            # incremental save so a Colab disconnect doesn't lose finished runs
            (out_dir / "results.json").write_text(json.dumps(results, indent=2))

            # Track the "best" run for the plot: prefer high accuracy AND high
            # sparsity. A simple combined score works well here.
            score = r.test_accuracy * (1.0 + r.sparsity_pct / 100.0)
            if best is None or score > best[0]:
                best = (score, model, label, r)

        _, best_model, best_label, best_r = best
        plot_gate_distribution(
            best_model,
            out_dir / "gate_distribution.png",
            title=f"Gate distribution -- best run: {best_label} "
                  f"(acc={best_r.test_accuracy:.1f}%, sparsity={best_r.sparsity_pct:.1f}%)",
        )
        print("\nall configurations done. results in:", out_dir / "results.json")
    else:
        model, r = train_one_config(
            lam=args.lam, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, subset=args.subset, device=device, seed=args.seed,
        )
        (out_dir / "results.json").write_text(json.dumps([{"label": "single", **asdict(r)}], indent=2))
        plot_gate_distribution(
            model,
            out_dir / "gate_distribution.png",
            title=f"Gate distribution (lambda={args.lam:g}, "
                  f"acc={r.test_accuracy:.1f}%, sparsity={r.sparsity_pct:.1f}%)",
        )


if __name__ == "__main__":
    main()
