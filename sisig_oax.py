"""SISig / Supplementary Algorithm S1: Orthogonal Array Crossover (OAX).

Aligned to ``Supplementary_Algorithms.docx`` (2026-11-16 upload), S1.
A and B are length-|P| binary chromosomes with exactly ``s_target`` ones.
``fitness(H, D)`` returns a scalar to MAXIMIZE. The article uses negative
cross-validated MAE for SVR in its mUC implementation.

Scope: crossover operator ONLY. The outer OAX-based evolutionary algorithm
(OAX-EA), mutation, and SIRE are described by Supplementary Algorithm S2 but
are NOT included in this standalone repository. The feature-level MED of S2
must NOT be confused with this module's segment-level MED.

Explicit conventions not specified by S1:
* Contiguous segments preserve original P order. Identical loci may be attached
  to an adjacent balanced segment. We use the earliest available balanced cut.
* Exact main-effect ties favor level 1 (A); minimum segment MED ties favor the
  smallest segment index (original P order).
* The first M nonzero parity columns define the canonical two-level L_q.
  Other valid OA column assignments can give different results for nonlinear
  fitness functions; this implementation fixes the assignment reproducibly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Any
import numpy as np


@dataclass(frozen=True)
class OAXResult:
    C1: np.ndarray
    C2: np.ndarray
    segments: Tuple[Tuple[int, int], ...]
    L_q: np.ndarray
    trial_fitness: np.ndarray
    ME_level_1: np.ndarray
    ME_level_2: np.ndarray
    MED_seg: np.ndarray
    flipped_segment: Optional[int]

    @property
    def M(self) -> int:
        return len(self.segments)

    @property
    def q(self) -> int:
        return len(self.trial_fitness)

    @property
    def child1(self) -> np.ndarray:  # legacy compatibility
        return self.C1

    @property
    def child2(self) -> np.ndarray:  # legacy compatibility
        return self.C2

    @property
    def oa(self) -> np.ndarray:  # legacy compatibility
        return self.L_q

    @property
    def me_level_1(self) -> np.ndarray:
        return self.ME_level_1

    @property
    def me_level_2(self) -> np.ndarray:
        return self.ME_level_2

    @property
    def segment_med(self) -> np.ndarray:
        return self.MED_seg

    @property
    def factor_count(self) -> int:
        return self.M

    @property
    def oa_trial_count(self) -> int:
        return self.q


def binary_chromosome(values: Any, label: str = 'chromosome') -> np.ndarray:
    p = np.asarray(values)
    if p.ndim != 1 or p.size == 0:
        raise ValueError(f'{label} must be a nonempty one-dimensional binary chromosome')
    if not np.isin(p, (0, 1)).all():
        raise ValueError(f'{label} must contain only 0/1 values')
    return np.array(p, dtype=np.int8, copy=True)


def maximal_balanced_segments(A: np.ndarray, B: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    """S1/1: maximum number of nonidentical balanced contiguous paired segments.

    Returns Python half-open ranges [start, end). ``M=0`` means A == B.
    """
    a, b = binary_chromosome(A, 'A'), binary_chromosome(B, 'B')
    if a.shape != b.shape or int(a.sum()) != int(b.sum()):
        raise ValueError('A and B must have the same feature pool and s_target')
    if np.array_equal(a, b):
        return ()
    cuts = [0]
    balance = 0
    for i, (av, bv) in enumerate(zip(a, b)):
        balance += int(av) - int(bv)
        if av != bv and balance == 0:
            cuts.append(i + 1)
    if balance != 0 or len(cuts) < 2:
        raise AssertionError('feasible parents must end at a valid balanced cut')
    # Any common trailing positions belong to the last nonidentical segment.
    cuts[-1] = len(a)
    out = tuple((int(u), int(v)) for u, v in zip(cuts[:-1], cuts[1:]))
    assert all(a[u:v].sum() == b[u:v].sum() and not np.array_equal(a[u:v], b[u:v])
               for u, v in out)
    return out


def two_level_orthogonal_array(M: int) -> np.ndarray:
    """S1/2: canonical L_q, q = smallest 2**k with q-1 >= M, first M columns.

    All q rows are used. For M=1, L_2 is a balanced two-level design;
    pairwise strength 2 is only meaningful when M >= 2.
    Encoded values: 0 = level 1 (A), 1 = level 2 (B).
    """
    if not isinstance(M, (int, np.integer)) or M < 0:
        raise ValueError('M must be a nonnegative integer')
    if M == 0:
        return np.empty((0, 0), dtype=np.int8)
    q = 2
    while q - 1 < M:
        q *= 2
    L_q = np.empty((q, M), dtype=np.int8)
    for row in range(q):
        for col in range(M):
            L_q[row, col] = (row & (col + 1)).bit_count() & 1
    return L_q


def compose_segments(A: np.ndarray, B: np.ndarray,
                     segments: Tuple[Tuple[int, int], ...], levels: np.ndarray) -> np.ndarray:
    c = A.copy()
    for (start, stop), level in zip(segments, levels):
        if level == 1:
            c[start:stop] = B[start:stop]
    return c


class OrthogonalArrayCrossover:
    """S1-only OAX operator, not OAX-EA or SIRE.

    ``crossover(A, B, D, fitness)`` reproduces S1's four named inputs.
    The method returns C1 and C2 AND transparent intermediate OA/MED results.
    C1/C2 fitness is intentionally not fabricated: S2/OAX-EA evaluates the
    offspring after mutation, as specified by S2 lines 81-84.
    """
    def crossover(self, A, B, D, fitness: Callable[[np.ndarray, Any], float]) -> OAXResult:
        a, b = binary_chromosome(A, 'A'), binary_chromosome(B, 'B')
        if a.shape != b.shape or int(a.sum()) != int(b.sum()):
            raise ValueError('A and B must be equally long and have the same s_target')
        if not callable(fitness):
            raise TypeError('fitness must be a callable fitness(chromosome, D)')
        segments = maximal_balanced_segments(a, b)  # S1/1
        M = len(segments)
        if M == 0:
            return OAXResult(a, b, (), np.empty((0, 0), dtype=np.int8),
                             np.empty(0, dtype=float), np.empty(0, dtype=float),
                             np.empty(0, dtype=float), np.empty(0, dtype=float), None)
        L_q = two_level_orthogonal_array(M)  # S1/2
        trial_fitness = np.empty(len(L_q), dtype=float)
        for t, levels in enumerate(L_q):  # S1/3
            H_t = compose_segments(a, b, segments, levels)
            score = float(fitness(H_t.copy(), D))
            if not np.isfinite(score):
                raise ValueError(f'nonfinite fitness at S1 orthogonal trial {t}')
            trial_fitness[t] = score
        ME_level_1 = np.array([trial_fitness[L_q[:, m] == 0].mean()
                               for m in range(M)], dtype=float)  # S1/4
        ME_level_2 = np.array([trial_fitness[L_q[:, m] == 1].mean()
                               for m in range(M)], dtype=float)
        MED_seg = np.abs(ME_level_1 - ME_level_2)
        preferred = (ME_level_2 > ME_level_1).astype(np.int8)  # S1/5, tie -> A
        C1 = compose_segments(a, b, segments, preferred)
        flip = int(np.argmin(MED_seg))  # S1/6, tie -> earliest segment
        alt = preferred.copy()
        alt[flip] ^= 1
        C2 = compose_segments(a, b, segments, alt)
        if int(C1.sum()) != int(a.sum()) or int(C2.sum()) != int(a.sum()):
            raise AssertionError('S1 returned invalid fixed-cardinality offspring')
        if np.array_equal(C1, C2):
            raise AssertionError('S1 requires C1 != C2 whenever M > 0')
        return OAXResult(C1, C2, segments, L_q, trial_fitness,
                         ME_level_1, ME_level_2, MED_seg, flip)  # S1/7


# Legacy standalone API compatibility; new code should use OrthogonalArrayCrossover.
class AlgorithmS1OAX(OrthogonalArrayCrossover):
    def crossover(self, parent_a, parent_b, fitness):
        """Historical API: fitness(mask), with discovery D captured by caller."""
        return super().crossover(parent_a, parent_b, None,
                                 lambda chromosome, _D: fitness(chromosome))

smallest_orthogonal_array = two_level_orthogonal_array
