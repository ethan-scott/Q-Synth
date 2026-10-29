"""
Faster search for the same optimum as chain_encoding.solve. Three changes, none of them in the specification:
  - one formula: slice j gets K_j steps, so one formula serves every split; "slice j uses at most k_j CNOTs" is
    the assumption "step k_j of slice j is empty";
  - lower bounds: every run of up to `window` consecutive slices is solved on its own; a split that gives a run
    fewer CNOTs than its bound is skipped;
  - cores: an unsatisfiable split names the slices whose limits caused it; every split that is at most as large on
    those slices is skipped.
All three rely on C3a allowing empty steps: a split that is satisfiable stays satisfiable when any k_j grows.
"""
from pysat.solvers import Solver

from qsynth.ReachabilitySolver.encodings.chain_synthesis import chain_encoding
from qsynth.ReachabilitySolver.encodings.chain_synthesis.chain_encoding import ChainEncoding


def column_bound(goal_matrix):
    """Every non-unit column needs a CNOT, and a column with w ones needs >= w - 1 CNOTs."""
    weights = goal_matrix.sum(axis=0)
    return int(max((weights > 1).sum(), weights.max() - 1))


def partition_bounds(bounds, s):
    """Best total lower bound for slices 1..s from the run bounds, and for each j the same without slice j."""
    prefix, suffix = {0: 0}, partition_suffix(bounds, s)
    for j in range(1, s + 1):
        prefix[j] = max(prefix[a - 1] + bounds[(a, j)] for a in range(1, j + 1) if (a, j) in bounds)
    return prefix[s], {j: prefix[j - 1] + suffix[j + 1] for j in range(1, s + 1)}


def partition_suffix(bounds, s):
    """suffix[j] = best lower bound for slices j..s."""
    suffix = {s + 1: 0}
    for j in range(s, 0, -1):
        suffix[j] = max(bounds[(j, b)] + suffix[b + 1] for b in range(j, s + 1) if (j, b) in bounds)
    return suffix


def splits(s, B, bounds, K, cores):
    """Splits (k_1..k_s) of B with k_j <= K_j that meet every run bound and are not ruled out by a core."""
    k, suffix = [0] * (s + 1), partition_suffix(bounds, s)

    def extend(j, left):
        if j > s:
            if left == 0:
                yield tuple(k[1:])
            return
        for v in range(bounds[(j, j)], min(K[j], left - suffix[j + 1]) + 1):
            k[j] = v
            if any((a, j) in bounds and sum(k[a:j + 1]) < bounds[(a, j)] for a in range(1, j)):
                continue
            if any(max(core) <= j and all(k[i] <= c for i, c in core.items()) for core in cores):
                continue
            yield from extend(j + 1, left - v)
    yield from extend(1, B)


def search(num_physical_qubits, edges, goal_matrices, bounds):
    """Smallest total CNOT count, one solver per slot budget. Returns (cost, layouts, cnots)."""
    s = len(goal_matrices)
    lower, others = partition_bounds(bounds, s)
    B, slack = lower, 4
    while True:
        K = {j: lower + slack - others[j] for j in goal_matrices}
        encoding, cores = ChainEncoding(num_physical_qubits, edges, goal_matrices, K), []
        with Solver(name="cd19") as solver:
            solver.append_formula(encoding.clauses)
            while B <= lower + slack:
                for k in splits(s, B, bounds, K, cores):
                    limits = {j: encoding.empty(j, k[j - 1]) for j in goal_matrices if k[j - 1] < K[j]}
                    if solver.solve(assumptions=[lit for lits in limits.values() for lit in lits]):
                        layouts, cnots = encoding.decode(solver.get_model())
                        return sum(map(len, cnots.values())), layouts, cnots
                    core = set(solver.get_core())
                    cores.append({j: k[j - 1] for j, lits in limits.items() if core & set(lits)})
                    if not cores[-1]:
                        break                      # unsatisfiable even without limits: more slots needed
                else:
                    B += 1
                    continue
                break
        slack += 4


def lower_bounds(num_physical_qubits, edges, goal_matrices, window=2):
    """bounds[(a, b)] = optimum of slices a..b solved on their own, for runs of up to `window` slices."""
    s, cache = len(goal_matrices), {}
    bounds = {(j, j): column_bound(goal_matrices[j]) for j in goal_matrices}
    for width in range(1, window + 1):
        for a in range(1, s - width + 2):
            b = a + width - 1
            key = tuple(goal_matrices[i].tobytes() for i in range(a, b + 1))
            if key not in cache:
                run = {i - a + 1: goal_matrices[i] for i in range(a, b + 1)}
                run_bounds = {(x - a + 1, y - a + 1): v for (x, y), v in bounds.items() if a <= x and y <= b}
                cache[key] = search(num_physical_qubits, edges, run, run_bounds)[0]
            bounds[(a, b)] = cache[key]
    return bounds


def solve(num_physical_qubits, edges, goal_matrices, window=2):
    return search(num_physical_qubits, edges, goal_matrices, lower_bounds(num_physical_qubits, edges, goal_matrices, window))[1:]


def chain_synthesis(circuit, coupling_graph, num_physical_qubits):
    return chain_encoding.chain_synthesis(circuit, coupling_graph, num_physical_qubits, solve=solve)
