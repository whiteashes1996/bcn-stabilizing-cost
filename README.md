# Discounted Stabilizing Cost for Boolean Control Networks

Companion code for the paper on computing the **discounted minimum stabilizing cost** for Boolean control networks (BCNs): the master pipeline (reachability restriction, condensation, folding, label-correcting on benign SCCs, beneficial-cycle detection, `SolveCore`, and feedback recovery).

## Repository layout

```
.
├── methods.py          # Reference Python library (exact Fraction / float64)
├── core_solver.py      # Earlier solver used by exp1–exp13
├── exp14_ex1_n4.py     # Example 1: 16-state full pipeline (exact arithmetic)
├── exp16_ex2_real.py   # Example 2: six literature networks
├── exp17_*.py          # Example 3: C++ ↔ Python cross-check and plots
├── cpp/                # C++ port of the pipeline (scalability)
├── networks/           # Literature network transition tables
├── results/            # Cached numerical outputs used in the paper
└── requirements.txt
```

## Requirements

- Python 3.11+ (paper timings: Python 3.14 on Apple M4)
- C++20 compiler (`clang++` or `g++`) for `cpp/`
- Optional: CBC via PuLP for the MILP oracle in `exp16`

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# Example 1 (exact rationals; writes results/exp14_ex1_n4.*)
python exp14_ex1_n4.py

# Example 2 (literature networks; float64)
python exp16_ex2_real.py

# C++ build and a short smoke run
cd cpp && make && ./build/bcn --help
```

Legacy scripts `exp1`–`exp13` and `run_all.py` remain for earlier drafts; the paper’s Illustrative Examples section corresponds to **exp14 / exp16 / exp17**.

## Citation

If you use this code, please cite the accompanying paper (link TBA).

## License

MIT
