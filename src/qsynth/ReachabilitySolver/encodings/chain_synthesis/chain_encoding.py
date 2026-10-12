import itertools

import numpy as np
from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pysat.solvers import Solver
from qiskit import QuantumCircuit


class ChainEncoding:
    def __init__(self, n, edges, M, k):
        self.n, self.edges, self.M, self.k, self.s = n, edges, M, k, len(M)
        self.pool, self.clauses = IDPool(), []
        add, E, R = self.clauses.append, range(len(edges)), range(n)

        # C1: every boundary is a permutation (each value on one wire, each wire holds one value)
        for j in range(self.s + 1):
            for a in R:
                self.exactly_one([self.pi(j, a, p) for p in R])
                self.exactly_one([self.pi(j, x, a) for x in R])

        # C2: section j starts in layout pi^{j-1} (in m())

        for j in M:
            for t in range(k[j]):
                # C3a: exactly one CNOT per step, only on coupling-graph edges
                self.exactly_one([self.cnot(j, t, e) for e in E])
                # C3b: trg(q) <-> some CNOT has target q
                for q in R:
                    into_q = [self.cnot(j, t, e) for e in E if edges[e][1] == q]
                    add([-self.trg(j, t, q)] + into_q)
                    for lit in into_q:
                        add([-lit, self.trg(j, t, q)])
                for r in R:
                    for e, (p, q) in enumerate(edges):
                        c, ctrl, old, new = self.cnot(j, t, e), self.m(j, t, r, p), self.m(j, t, r, q), self.m(j, t + 1, r, q)
                        # C3c: control contains r -> target flips r
                        add([-c, -ctrl, old, new]); add([-c, -ctrl, -old, -new])
                        # C3d: control lacks r -> target keeps r
                        add([-c, ctrl, old, -new]); add([-c, ctrl, -old, new])
                    # C3e: a wire that is not the target stays the same
                    for q in R:
                        add([self.trg(j, t, q), -self.m(j, t, r, q), self.m(j, t + 1, r, q)])
                        add([self.trg(j, t, q), self.m(j, t, r, q), -self.m(j, t + 1, r, q)])
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
    def trg(self, j, t, q): return self.pool.id(("trg", j, t, q))

    def exactly_one(self, lits):
        self.clauses.extend(CardEnc.equals(lits=lits, bound=1, vpool=self.pool, encoding=EncType.seqcounter).clauses)

    def decode(self, model):
        """Returns (layouts pi^0..pi^s as {value: wire}, CNOTs per section as [(control, target)])."""
        true, R = {lit for lit in model if lit > 0}, range(self.n)
        layouts = [{x: p for x in R for p in R if self.pi(j, x, p) in true} for j in range(self.s + 1)]
        cnots = {j: [self.edges[e] for t in range(self.k[j]) for e in range(len(self.edges))
                     if self.cnot(j, t, e) in true] for j in self.M}
        return layouts, cnots


def slice_circuit(circuit):
    """Splits the circuit at its single-qubit gates into (one_qubit_blocks, cnot_blocks)."""
    one_qubit, cnots, current, pending, blocked = [[]], [], [], [], set()
    for instruction in circuit.data:
        name, qubits = instruction.operation.name, [circuit.find_bit(q).index for q in instruction.qubits]
        if name == "barrier":
            continue
        if name == "cx":
            if blocked & set(qubits):
                cnots.append(current); one_qubit.append(pending)
                current, pending, blocked = [], [], set()
            current.append(tuple(qubits))
        elif len(qubits) == 1:
            if current:
                pending.append((instruction.operation, qubits[0])); blocked.add(qubits[0])
            else:
                one_qubit[-1].append((instruction.operation, qubits[0]))
        else:
            raise ValueError(f"only cx and 1-qubit gates are supported, got {name}")
    if current:
        cnots.append(current); one_qubit.append(pending)
    return one_qubit, cnots


def parity_matrix(cnots, n):
    """Column x lists the input values XOR-ed together on qubit x after the CNOTs."""
    M = np.eye(n, dtype=int)
    for c, t in cnots:
        M[:, t] ^= M[:, c]
    return M


def solve(n, edges, M):
    """Smallest k_1 + ... + k_s for which C1-C4 are satisfiable: try the totals B = 0, 1, 2, ... and, for each B,
    every split (k_1, ..., k_s) of B. Returns (B, layouts, cnots)."""
    for B in itertools.count():
        for split in itertools.product(range(B + 1), repeat=len(M)):
            if sum(split) != B:
                continue
            encoding = ChainEncoding(n, edges, M, dict(zip(M, split)))
            with Solver(name="cd19", bootstrap_with=encoding.clauses) as solver:
                if solver.solve():
                    return (B,) + encoding.decode(solver.get_model())


def chain_synthesis(circuit, coupling_graph, n):
    """Maps `circuit` onto n wires with the given coupling graph using the fewest CNOTs (for this slicing).
    Returns (mapped circuit, CNOT count, initial mapping, final mapping); mappings are {logical qubit: wire}."""
    edges = sorted({(a, b) for a, b in coupling_graph} | {(b, a) for a, b in coupling_graph})
    one_qubit, blocks = slice_circuit(circuit)
    M = {j: parity_matrix(block, n) for j, block in enumerate(blocks, start=1)}
    if not M:
        return circuit.copy(), 0, {q: q for q in range(circuit.num_qubits)}, {q: q for q in range(circuit.num_qubits)}
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


if __name__ == "__main__":
    # cx q0,q2 . t q2 . cx q0,q1 on the line 0 - 1 - 2
    from qsynth.api import check_equivalence
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2); circuit.t(2); circuit.cx(0, 1)
    mapped, cost, initial, final = chain_synthesis(circuit, [(0, 1), (1, 2)], 3)
    check_equivalence(circuit, mapped, final_mapping=final, initial_mapping=initial)
    print(f"{cost} CNOTs, initial mapping {initial}, final mapping {final}, equivalent")
    print(mapped)
