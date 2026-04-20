# Self-Pruning Neural Network

Tredence Analytics — AI Engineering Intern case study.

A PyTorch image classifier on CIFAR-10 that **learns which of its own
connections to remove during training**, instead of pruning as a separate
post-training step.

The full write-up with results and the gate-distribution plot is in
**[REPORT.md](./REPORT.md)**.

If you have **no prior deep-learning background**, start with
**[WALKTHROUGH.md](./WALKTHROUGH.md)** — a top-to-bottom plain-English
explanation of the entire project, with a glossary at the end.

---

## What problem are we solving?

Modern neural networks are huge — millions or billions of weights. Most of
those weights end up barely contributing to the answer. **Pruning** means
deleting the unimportant ones to make the model smaller and faster. The usual
recipe is: train normally, then prune.

This project does it differently. We let the network **decide for itself, while
it's training, which connections to keep**.

## How it works

1. **The PrunableLinear layer.** A normal linear layer in PyTorch
   (`nn.Linear`) holds a matrix of `weights` and a `bias`. Our custom
   `PrunableLinear` holds a third tensor of the same shape as `weights`,
   called `gate_scores`. Every `gate_score` becomes a **gate** in the range
   `[0, 1]` after passing through a sigmoid. The actual computation in the
   forward pass uses `weight * gate` instead of just `weight`. If a gate is
   close to 0, that connection is effectively turned off.

2. **The gates are learnable.** `gate_scores` is registered as a parameter,
   so the optimizer (Adam in our case) updates them just like the weights.
   The network learns *which gates to close*.

3. **A penalty pushes gates toward 0.** Without pressure, the optimizer would
   leave all gates open. So we add a term to the loss:

       total_loss = cross_entropy + lambda * sum(all gates)

   The bigger `lambda`, the harder the optimizer is pushed to close gates.
   Cross-entropy keeps the gates open *only where they actually help with
   classification accuracy*.

4. **At evaluation time, count the closed gates.** A gate below `1e-2` is
   considered "pruned". The percentage of pruned gates is the **sparsity
   level**.

## Repo layout

```
self-pruning-nn/
├── self_pruning.py     # the single, well-commented script the brief asks for
├── requirements.txt    # torch, torchvision, matplotlib, numpy
├── README.md           # this file
├── REPORT.md           # explanation + results table + plot (the deliverable)
└── outputs/            # written by the script
    ├── results.json
    └── gate_distribution.png
```

## How to run

### Locally (or in a venv)

```bash
pip install -r requirements.txt

# Quick smoke test (1 epoch on a tiny subset, ~1 min on CPU):
python self_pruning.py --lam 1e-6 --epochs 1 --subset 1024

# Single configuration with a specific lambda:
python self_pruning.py --lam 5e-6 --epochs 25

# All four configurations from REPORT.md (baseline + 3 lambdas):
python self_pruning.py --run-all --epochs 25
```

### Google Colab (T4 GPU, recommended for the full run)

```python
# In a Colab cell:
!git clone <your-repo-url>
%cd self-pruning-nn
!pip install -r requirements.txt
!python self_pruning.py --run-all --epochs 25
```

The full sweep on a T4 takes roughly 30–45 minutes and writes
`outputs/results.json` and `outputs/gate_distribution.png`.

## What to look at

After running `--run-all`:

- `outputs/results.json` — final accuracy and sparsity for every lambda.
- `outputs/gate_distribution.png` — histogram of all gate values for the
  best run. A successful pruning run looks **bimodal**: a tall spike near 0
  (pruned connections) and a smaller cluster near 1 (kept connections).
- `REPORT.md` — full write-up.

## Design choices worth defending in an interview

- **Why initialise `gate_scores` at +2.0?** `sigmoid(2.0) ≈ 0.88`. That keeps
  the network close to a normal MLP at step 0, so training starts stably.
  Initialising at 0 (gates = 0.5) halves the signal at init and slows
  convergence.
- **Why L1 instead of L2?** L1's gradient stays constant near 0, so the
  optimizer is willing to push gates *all the way* to 0. L2's gradient shrinks
  with the value, so it never quite reaches 0 — you'd get small-but-nonzero
  gates and no real sparsity.
- **Why `sum` of gates, not `mean`?** The brief asks for the L1 norm. We use
  sum and pick `lambda` accordingly (small values like 5e-6 because the sum
  is ~3.3 million at init).
- **Why a plain MLP and no data augmentation?** The case study is grading
  **the pruning mechanism**, not raw accuracy. A simpler model with no
  augmentation makes it cleaner to attribute gains/losses to pruning.
- **Built-in gradient-flow check.** The script asserts at startup that
  `gate_scores` actually receives a non-zero gradient — the most common bug
  in custom layers like this is forgetting to register a tensor as a
  parameter. The brief explicitly calls out gradient flow as the challenge.
