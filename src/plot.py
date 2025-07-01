"""Privacy / accuracy plotting helpers.

Two canonical plots for any DP-SGD run:
  1. Privacy curve: ε(δ) at fixed δ as a function of training step
  2. Tradeoff: accuracy vs. ε at the end of training across runs

We don't auto-import matplotlib at module load — only inside the plot
functions — so importing this module on a headless machine is fine.
"""
import json
import os
from typing import List, Tuple


def plot_privacy_curve(history: List[Tuple[int, float, float]], out_path: str):
    """history is a list of (step, eps, train_loss) tuples."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = [h[0] for h in history]
    epss = [h[1] for h in history]
    losses = [h[2] for h in history]

    fig, ax_eps = plt.subplots()
    ax_eps.plot(steps, epss, label='ε', color='C0')
    ax_eps.set_xlabel('Step')
    ax_eps.set_ylabel('ε', color='C0')
    ax_eps.tick_params(axis='y', labelcolor='C0')
    ax_eps.grid(alpha=0.3)

    ax_loss = ax_eps.twinx()
    ax_loss.plot(steps, losses, label='train loss', color='C1', alpha=0.6)
    ax_loss.set_ylabel('Loss', color='C1')
    ax_loss.tick_params(axis='y', labelcolor='C1')

    fig.suptitle('DP-SGD privacy budget growth + training loss')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tradeoff(runs: List[dict], out_path: str):
    """runs is a list of {'name': str, 'epsilon': float, 'accuracy': float}."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for r in runs:
        ax.scatter(r['epsilon'], r['accuracy'], s=80,
                   label=r.get('name', f"ε={r['epsilon']}"))
        ax.annotate(r.get('name', ''), (r['epsilon'], r['accuracy']),
                    textcoords='offset points', xytext=(5, 5))
    ax.set_xlabel('ε at end of training (δ=1e-5)')
    ax.set_ylabel('Test accuracy')
    ax.set_title('Privacy / accuracy tradeoff')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize_runs(history_jsons: List[str]) -> List[dict]:
    """Read multiple history JSON files and produce a runs list for plotting."""
    runs = []
    for path in history_jsons:
        with open(path) as f:
            h = json.load(f)
        runs.append({
            'name': h.get('name', os.path.basename(path)),
            'epsilon': h['final_epsilon'],
            'accuracy': h['final_test_accuracy'],
        })
    return runs
