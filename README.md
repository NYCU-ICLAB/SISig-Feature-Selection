# SISig-OAX

**Standalone Orthogonal Array Crossover (OAX) operator for the SISig framework**

This repository provides a minimal Python implementation of **Orthogonal Array Crossover (OAX)** as specified in **Supplementary Algorithm S1** of the SISig manuscript:

> *SISig: A Stability-Oriented Framework for Robust Immunotherapy Signature Discovery in Cancer Transcriptomics*

SISig (*Stable Immunotherapy Signature*) combines knowledge-based constraints (KBC) and **Signature Iterative Refinement with Ensembles (SIRE)**. This repository contains **only the OAX crossover operator**, which is a component of the **OAX-based evolutionary algorithm (OAX-EA)** used within SIRE. It does **not** implement the full OAX-EA, SIRE, feature-level contributor analysis, or the complete immunotherapy prediction pipeline.

## Repository contents

```text
SISig-OAX/
├── README.md          # Documentation and runnable example
├── sisig_oax.py       # Supplementary Algorithm S1 — OAX only
├── requirements.txt   # NumPy dependency
└── .gitignore         # Local/generated files to ignore
```

## Requirements and installation

- **Python 3.10 or later** (`int.bit_count()` is used to construct the orthogonal array).
- **NumPy** (see `requirements.txt`).

Clone the repository and install the dependency:

```bash
git clone https://github.com/NYCU-ICLAB/SISig-Feature-Selection.git
cd SISig-OAX
python -m pip install -r requirements.txt
```

This command assumes that the repository is created under **`vincentyan7461`** with the name **`SISig-OAX`**. If you use a different owner or repository name, update the clone URL. The module can also be imported directly after downloading `sisig_oax.py` and installing NumPy; packaging or installation of the module is not required.

## Algorithm overview (Supplementary Algorithm S1)

The input consists of two feasible binary parent chromosomes **A** and **B** over the same ordered candidate feature pool **P**, each containing exactly `s_target` selected features (1-bits), a training dataset **D**, and a fitness function `fitness(chromosome, D)` for which **higher is better**.

OAX performs the following seven steps:

1. **Segment partitioning.** Partition A and B into the maximal number **M** of corresponding, nonidentical, balanced segments, where each paired segment has the same number of selected features. Return A and B unchanged if `M = 0`.
2. **Orthogonal array construction.** Choose **q** as the smallest power of two for which `q - 1 >= M`. Construct a two-level orthogonal array **L_q** using the first M columns. Each segment is a factor; level 1 copies parent A's segment and level 2 copies parent B's segment.
3. **Orthogonal trial evaluation.** For every row **t** of L_q, construct a trial chromosome **H_t** and evaluate `fitness(H_t, D)`.
4. **Segment-level main effects.** Compute the average fitness under level 1 and level 2 for each segment m: `ME_m,1` and `ME_m,2`. Compute `MED_m^seg = |ME_m,1 - ME_m,2|`.
5. **First offspring (C1).** For each segment, choose the parent level with the larger main effect.
6. **Second offspring (C2).** Start with C1's segment-level choices and flip the level of the segment with the smallest `MED_m^seg`; break ties using a fixed deterministic order.
7. **Return.** Return feasible children **C1** and **C2**. Both retain exactly `s_target` selected features.

This is **segment-level** analysis inside the crossover. It must not be confused with **feature-level** main-effect contrasts, `d_j` and `MED_j`, used for contributor retention in **Supplementary Algorithm S2 (SIRE)**.

## Quick start: runnable example

The example below uses a **synthetic weighted-sum fitness function**, not an SVR model or clinical data. Save it as `demo.py` in the repository folder, or paste it into a Python session started in that folder.

```python
import numpy as np
from sisig_oax import OrthogonalArrayCrossover

# Parents have the same feature-pool order and exactly three selected features.
A = np.array([1, 0, 1, 0, 1, 0], dtype=np.int8)
B = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)

# D can be any object consumed by the user-defined fitness function.
D = {"weights": np.array([6, 1, 4, 2, 5, 3], dtype=float)}

def fitness(chromosome: np.ndarray, D: dict) -> float:
    return float(chromosome @ D["weights"])  # Larger is better.

result = OrthogonalArrayCrossover().crossover(A, B, D, fitness)

print("Segments:", result.segments)
print("M:", result.M, "q:", result.q)
print("C1:", result.C1.tolist())
print("C2:", result.C2.tolist())
print("MED_seg:", result.MED_seg.tolist())
print("Selected-feature counts:", int(result.C1.sum()), int(result.C2.sum()))
```

Expected output:

```text
Segments: ((0, 2), (2, 4), (4, 6))
M: 3 q: 4
C1: [1, 0, 1, 0, 1, 0]
C2: [1, 0, 0, 1, 1, 0]
MED_seg: [5.0, 2.0, 2.0]
Selected-feature counts: 3 3
```

## API

```python
from sisig_oax import OrthogonalArrayCrossover
result = OrthogonalArrayCrossover().crossover(A, B, D, fitness)
```

| Name | Meaning |
| --- | --- |
| `A`, `B` | Nonempty, one-dimensional 0/1 chromosomes with equal lengths and equal numbers of 1-bits; the feature order must match. |
| `D` | Dataset or user-defined context passed unmodified to `fitness`; it can be any Python object. |
| `fitness(mask, D)` | Callable returning a **finite scalar** to **maximize**. This repository does not prescribe or train a prediction model. |
| `result.C1`, `result.C2` | NumPy arrays containing the two feasible offspring. |
| `result.segments` | Balanced segment ranges as zero-based, half-open tuples `(start, stop)`. |
| `result.M`, `result.q` | Number of segment factors and number of orthogonal trials. |
| `result.L_q` | Two-level orthogonal design; `0` means level 1 (parent A), `1` means level 2 (parent B). |
| `result.trial_fitness` | Fitness values for the **q orthogonal trials**. |
| `result.ME_level_1`, `result.ME_level_2` | Per-segment conditional mean fitness at each parent level. |
| `result.MED_seg` | Per-segment absolute main-effect differences. |
| `result.flipped_segment` | Zero-based index of the segment whose level was flipped to construct C2; `None` when `M = 0`. |

The result type is `OAXResult`. Historical aliases (e.g., `child1`, `child2`, and `AlgorithmS1OAX`) are retained in the source for backward compatibility; **`OrthogonalArrayCrossover` and `C1`/`C2` are the preferred interface**.

### Fitness in the SISig manuscript

The SISig case study used **support vector regression (SVR)** with **cross-validated negative mean absolute error (negative MAE)** as the optimization objective. To use this OAX implementation in that context, the caller must supply its own negative-MAE fitness function, cross-validation protocol, preprocessing, and model settings. These are **not included** in this repository. The synthetic example above does not reproduce manuscript experiments or reported performance.

### Reproducibility and implementation conventions

Supplementary Algorithm S1 leaves some implementation details open. This module uses the following explicit, deterministic conventions:

- Segment boundaries are sought in the original feature-pool order. A segment ends at the earliest feasible running-balance return to zero; common genes are attached to an adjacent segment as needed.
- For an exact main-effect tie, level 1 (parent A) is preferred.
- If several segments share the minimum `MED_seg`, the lowest segment index is flipped.
- The first **M** nonzero parity columns of the canonical two-level `L_q` are used. Different valid column assignments can produce different effects for nonlinear fitness functions.
- If `A == B`, `M = 0`: the parents are returned as `C1` and `C2` without fitness calls.

The method calls the fitness function **q times** when `M > 0`. It does **not** calculate fitness for C1 and C2 after crossover; any enclosing evolutionary algorithm should score the offspring using its own evaluation protocol (including after mutation when applicable). Estimated main effects do not guarantee that C1 is globally optimal.

## Scope and limitations

- **Included:** the standalone S1 OAX operator, transparent intermediate results, equal-cardinality checks, and deterministic tie handling.
- **Not included:** population initialization, parent selection, mutation, survivor selection, convergence logic, feature-level contributor analysis, SIRE candidate-pool updates, ensemble fitting, preprocessing, trained models, or patient-level data.
- The operator assumes valid fixed-cardinality parent feature masks. A complete OAX-EA/SIRE implementation must manage feasibility and fitness at the surrounding search level.
- This implementation expresses the S1 procedure as documented; **publishing this module alone is not a full reproduction of the SISig study**.

## Reference

**Manuscript:** Li-Chen Cheng and Shinn-Ying Ho. *SISig: A Stability-Oriented Framework for Robust Immunotherapy Signature Discovery in Cancer Transcriptomics.* See **Supplementary Algorithm S1: Orthogonal Array Crossover procedure (OAX)**. Publication details and a DOI can be added when available.

## License and availability

No open-source license is included at this stage. A public GitHub repository makes the source **viewable**, but does not by itself grant broad reuse, redistribution, or modification rights. The repository owner can add a license later if desired.
