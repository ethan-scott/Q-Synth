import itertools

from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pysat.solvers import Solver
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import RemoveBarriers

from qsynth.ReachabilitySolver.encodings.cnot_synthesis.cnot_reachability_utils import get_goal_matrix_from_cnot_circuit
from qsynth.Utilities.slices import get_slices
from qsynth.api import make_bidirectional_graph


class ChainEncoding:
    def __init__(self, n, edges, M, k):
        self.n, self.edges, self.k = n, edges, k
        self.pool, self.clauses = IDPool(), []
        add, E, R = self.clauses.append, range(len(edges)), range(n)

        # C1: every boundary is a permutation (each value on one wire, each wire holds one value)
        for j in range(len(M) + 1):
            for a in R:
                self.exactly_one([self.pi(j, a, p) for p in R])
                self.exactly_one([self.pi(j, x, a) for x in R])

        # C2: section j starts in layout pi^{j-1} (in m())

        for j in M:
            for t in range(k[j]):
                # C3a: at most one CNOT per step, only on coupling-graph edges; empty steps come last
                self.at_most_one_cnot(j, t)
                into = {q: [self.cnot(j, t, e) for e in E if edges[e][1] == q] for q in R}
                for r in R:
                    for e, (p, q) in enumerate(edges):
                        c, ctrl, old, new = self.cnot(j, t, e), self.m(j, t, r, p), self.m(j, t, r, q), self.m(j, t + 1, r, q)
                        # C3b: control contains r -> target flips r
                        add([-c, -ctrl, old, new]); add([-c, -ctrl, -old, -new])
                        # C3c: control lacks r -> target keeps r
                        add([-c, ctrl, old, -new]); add([-c, ctrl, -old, new])
                    # C3d: a wire that changes is the target of this step's CNOT
                    for q in R:
                        add([-self.m(j, t, r, q), self.m(j, t + 1, r, q)] + into[q])
                        add([self.m(j, t, r, q), -self.m(j, t + 1, r, q)] + into[q])
            # C4: value x on wire p at boundary j -> wire p holds column x of M_j
            for x in R:
                for p in R:
                    for r in R:
                        add([-self.pi(j, x, p), self.m(j, k[j], r, p) if M[j][r, x] else -self.m(j, k[j], r, p)])

    def m(self, j, t, r, p):
        if t == 0:
            return self.pi(j - 1, r, p)          # C2
        return self.pool.id(("m", j, t, r, p))

    def pi(self, j, x, p): return self.pool.id(("pi", j, x, p))
    def cnot(self, j, t, e): return self.pool.id(("cnot", j, t, e))
    def step(self, j, t): return [self.cnot(j, t, e) for e in range(len(self.edges))]

    def at_most_one_cnot(self, j, t):
        self.clauses.extend(CardEnc.atmost(lits=self.step(j, t), bound=1, vpool=self.pool, encoding=EncType.seqcounter).clauses)
        if t > 0:
            for lit in self.step(j, t):
                self.clauses.append([-lit] + self.step(j, t - 1))

    def empty(self, j, t):
        return [-lit for lit in self.step(j, t)]

    def exactly_one(self, lits):
        self.clauses.extend(CardEnc.equals(lits=lits, bound=1, vpool=self.pool, encoding=EncType.seqcounter).clauses)

    def decode(self, model):
        """Returns (layouts pi^0..pi^s as {value: wire}, CNOTs per section as [(control, target)])."""
        true, R = {lit for lit in model if lit > 0}, range(self.n)
        layouts = [{x: p for x in R for p in R if self.pi(j, x, p) in true} for j in range(len(self.k) + 1)]
        cnots = {j: [self.edges[e] for t in range(self.k[j]) for e in range(len(self.edges))
                     if self.cnot(j, t, e) in true] for j in self.k}
        return layouts, cnots


def solve(n, edges, M):
    """Smallest total CNOT count: for B = 0, 1, 2, ..., give every section B steps and allow at most B CNOTs in
    total. This is the smallest k_1 + ... + k_s for which C1-C4 are satisfiable. Returns (B, layouts, cnots)."""
    for B in itertools.count():
        encoding = ChainEncoding(n, edges, M, {j: B for j in M})
        every = [lit for j in M for t in range(B) for lit in encoding.step(j, t)]
        total = CardEnc.atmost(lits=every, bound=B, vpool=encoding.pool, encoding=EncType.totalizer).clauses
        with Solver(name="cd19", bootstrap_with=encoding.clauses + total) as solver:
            if solver.solve():
                return (B,) + encoding.decode(solver.get_model())


def chain_synthesis(circuit, coupling_graph, n, solve=solve):
    """Maps `circuit` onto n wires with the given coupling graph using the fewest CNOTs (for this slicing).
    Returns (mapped circuit, CNOT count, initial mapping, final mapping); mappings are {logical qubit: wire}."""
    edges = make_bidirectional_graph(coupling_graph)
    # Sections: Q-Synth's maximal CNOT slices (as in the ECAI'24 peephole); one_qubit[j] runs after section j
    circuit = RemoveBarriers()(circuit)
    for instruction in circuit.data:
        if instruction.operation.name != "cx" and len(instruction.qubits) != 1:
            raise ValueError(f"only cx and 1-qubit gates are supported, got {instruction.operation.name}")
    qubits = [[circuit.find_bit(q).index for q in instruction.qubits] for instruction in circuit.data]
    one_qubit, M = [[]], {}
    for single, cx in get_slices(circuit, lambda instruction: instruction.operation.name == "cx"):
        one_qubit[-1] += [(circuit.data[g].operation, qubits[g][0]) for g in single]
        if cx:
            section = QuantumCircuit(n)
            for g in cx:
                section.cx(*qubits[g])
            M[len(M) + 1] = get_goal_matrix_from_cnot_circuit(section); one_qubit.append([])
    cost, layouts, cnots = solve(n, edges, M)
    mapped = QuantumCircuit(n)
    for j in range(len(M) + 1):
        if j > 0:
            for c, t in cnots[j]:
                mapped.cx(c, t)
        for gate, q in one_qubit[j]:
            mapped.append(gate, [layouts[j][q]])
    logical = range(circuit.num_qubits)
    return mapped, cost, {q: layouts[0][q] for q in logical}, {q: layouts[-1][q] for q in logical}

