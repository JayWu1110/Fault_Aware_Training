"""One-shot entry point kept from the original project.

Trains one model and evaluates it fault-free and faulty, equivalent to::

    python train.py [train args]
    python evaluate.py --run outputs/<exp-name>

All ``train.py`` arguments are accepted, e.g.::

    python fat.py --mode fat --seed 0 --epochs 20
"""
import sys

import evaluate
import train


def main(argv=None):
    cfg = train.parse_args(argv)
    run_dir = train.train(cfg)
    evaluate.main(["--run", str(run_dir), "--probs", str(cfg.fault_prob),
                   "--fault-models", cfg.fault_model, "--repeats", "10"])


if __name__ == "__main__":
    main(sys.argv[1:])
