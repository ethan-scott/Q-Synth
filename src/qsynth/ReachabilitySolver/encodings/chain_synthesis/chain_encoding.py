import itertools

from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pysat.solvers import Solver
from qiskit import QuantumCircuit

from qsynth.PeepholeSlicing.circuit_utils import is_cnot_gate
from qsynth.ReachabilitySolver.encodings.cnot_synthesis.cnot_reachability_utils import get_goal_matrix_from_cnot_circuit
from qsynth.Utilities.result import MappingResult
from qsynth.Utilities.slices import get_slices
from qsynth.api import make_bidirectional_graph


class ChainEncoding:
    # Variables, as in ChainEncoding.pdf:
    #   pi^j_{x,p}     logical qubit x sits on physical qubit p at the end of slice j (pi^0: initial layout)
    #   ctrl^{j,t}_p   physical qubit p is the control of the CNOT at step t of slice j
    #   trg^{j,t}_q    physical qubit q is the target of the CNOT at step t of slice j
    #   m^{j,t}_{r,p}  the parity of physical qubit p includes logical qubit r after t CNOTs of slice j
    #   used^{j,t}     step t of slice j has a CNOT (the counter variable: B counts these, one per step)
    # Indices: j slice, t step, x and r logical qubits, p and q physical qubits.
    def __init__(self, num_physical_qubits, edges, goal_matrices, steps):
        self.num_physical_qubits, self.steps = num_physical_qubits, steps
        self.id_pool, self.clauses = IDPool(), []
        add_clause, qubits = self.clauses.append, range(num_physical_qubits)
        edge_set = set(map(tuple, edges))

        # C1: every pi^j is one-hot: each logical qubit on exactly one physical qubit and vice versa
        for j in range(len(goal_matrices) + 1):
            for x in qubits:
                self.exactly_one([self.pi(j, x, p) for p in qubits])
            for p in qubits:
                self.exactly_one([self.pi(j, x, p) for x in qubits])

        # C2: m^{j,0} = pi^{j-1} (in m())

        for j in goal_matrices:
            for t in range(steps[j]):
                # C3a: at most one control and one target per step; used^{j,t} iff the step has a control,
                # iff it has a target
                used = self.used(j, t)
                self.at_most_one([self.ctrl(j, t, p) for p in qubits])
                self.at_most_one([self.trg(j, t, q) for q in qubits])
                add_clause([-used] + [self.ctrl(j, t, p) for p in qubits])
                add_clause([-used] + [self.trg(j, t, q) for q in qubits])
                for p in qubits:
                    add_clause([-self.ctrl(j, t, p), used])
                    add_clause([-self.trg(j, t, p), used])
                # C3b: control and target are a coupling-graph edge (this also rules out p = q)
                for p in qubits:
                    for q in qubits:
                        if (p, q) not in edge_set:
                            add_clause([-self.ctrl(j, t, p), -self.trg(j, t, q)])
                # C3c: empty steps come last
                if t > 0:
                    add_clause([-used, self.used(j, t - 1)])
                for r in qubits:
                    # C3d: the target q flips r if the control p has r, and keeps r otherwise
                    for p, q in edge_set:
                        ctrl, trg = self.ctrl(j, t, p), self.trg(j, t, q)
                        control_has_r, target_had_r, target_has_r = self.m(j, t, r, p), self.m(j, t, r, q), self.m(j, t + 1, r, q)
                        add_clause([-ctrl, -trg, -control_has_r, target_had_r, target_has_r])
                        add_clause([-ctrl, -trg, -control_has_r, -target_had_r, -target_has_r])
                        add_clause([-ctrl, -trg, control_has_r, target_had_r, -target_has_r])
                        add_clause([-ctrl, -trg, control_has_r, -target_had_r, target_has_r])
                    # C3e: a physical qubit that is not the target keeps its parity
                    for q in qubits:
                        add_clause([self.trg(j, t, q), -self.m(j, t, r, q), self.m(j, t + 1, r, q)])
                        add_clause([self.trg(j, t, q), self.m(j, t, r, q), -self.m(j, t + 1, r, q)])
            # C4: m^{j,final} = M_j pi^j: if logical qubit x sits on p in pi^j, the parity of p is column x of M_j
            for x in qubits:
                for p in qubits:
                    for r in qubits:
                        final = self.m(j, steps[j], r, p)
                        add_clause([-self.pi(j, x, p), final if goal_matrices[j][r, x] else -final])

    def m(self, j, t, r, p):
        if t == 0:
            return self.pi(j - 1, r, p)          # C2
        return self.id_pool.id(("m", j, t, r, p))

    def pi(self, j, x, p): return self.id_pool.id(("pi", j, x, p))
    def ctrl(self, j, t, p): return self.id_pool.id(("ctrl", j, t, p))
    def trg(self, j, t, q): return self.id_pool.id(("trg", j, t, q))
    def used(self, j, t): return self.id_pool.id(("used", j, t))

    def empty(self, j, t):
        return [-self.used(j, t)]

    def at_most_one(self, lits):
        self.clauses.extend(CardEnc.atmost(lits=lits, bound=1, vpool=self.id_pool, encoding=EncType.seqcounter).clauses)

    def exactly_one(self, lits):
        self.clauses.extend(CardEnc.equals(lits=lits, bound=1, vpool=self.id_pool, encoding=EncType.seqcounter).clauses)

    def decode(self, model):
        """Returns (layouts pi^0..pi^s as {logical qubit: physical qubit}, CNOTs per slice as [(control, target)])."""
        true, qubits = {lit for lit in model if lit > 0}, range(self.num_physical_qubits)
        layouts = [{x: p for x in qubits for p in qubits if self.pi(j, x, p) in true} for j in range(len(self.steps) + 1)]
        cnots = {j: [(p, q) for t in range(self.steps[j]) for p in qubits for q in qubits
                     if self.ctrl(j, t, p) in true and self.trg(j, t, q) in true] for j in self.steps}
        return layouts, cnots


def solve(num_physical_qubits, edges, goal_matrices):
    # B = total CNOT count across entire circuit
    for B in itertools.count():
        # Generate constraints
        encoding = ChainEncoding(num_physical_qubits, edges, goal_matrices, {j: B for j in goal_matrices})

        # the counter variables of all steps of all slices: one true used per CNOT
        used_vars = [encoding.used(j, t) for j in goal_matrices for t in range(B)]

        at_most_B_cnots = CardEnc.atmost(lits=used_vars, bound=B, vpool=encoding.id_pool, encoding=EncType.totalizer).clauses
        with Solver(name="cd19") as solver:
            solver.append_formula(encoding.clauses + at_most_B_cnots)
            if solver.solve():
                return encoding.decode(solver.get_model())


def chain_synthesis(circuit, coupling_graph, num_physical_qubits, solve=solve):
    # num_physical_qubits: number of qubits on the hardware

    # Get the bidirectional graph that represents where CNOTs can be placed in the physical hardware.
    edges = make_bidirectional_graph(coupling_graph)

    # Get the qubit indices that each gate acts on
    # Example: cx 0,2; t 2; cx 0,1 would correspond to: qubits = [[0,2], [2], [0,1]]
    qubits = []
    for instruction in circuit.data:
        qubits.append([circuit.find_bit(q).index for q in instruction.qubits])

    # goal_matrices[j] is the goal (parity) matrix of CNOT slice j
    goal_matrices = {}

    # gates_1q[j]: 1-qubit gates applied after slice j, as (gate, qubit index). 
    # The 1q gates are put back after the right slices, after the solve. 
    # gates_1q[0] comes before slice 1, if there are no CNOTs on qubits before those 1q gates
    gates_1q = [[]]

    for slice_1q, slice_cx in get_slices(circuit, is_cnot_gate):
        for g in slice_1q:
            gates_1q[-1].append((circuit.data[g].operation, qubits[g][0]))
        if slice_cx:
            # get the goal (parity) matrix of this CNOT slice
            cnot_slice = QuantumCircuit(num_physical_qubits)
            for g in slice_cx:
                cnot_slice.append(circuit.data[g].operation, qubits[g])
            goal_matrices[len(goal_matrices) + 1] = get_goal_matrix_from_cnot_circuit(cnot_slice) # function is pre-existing
            gates_1q.append([])

    
    layouts, cnots = solve(num_physical_qubits, edges, goal_matrices)


    # Construct final "optimized" circuit in the correct format, after the solve is finished
    mapped = QuantumCircuit(num_physical_qubits)
    for j in range(len(goal_matrices) + 1):
        if j > 0:
            for c, t in cnots[j]:
                mapped.cx(c, t)
        for gate, q in gates_1q[j]:
            mapped.append(gate, [layouts[j][q]])

    
    initial_mapping = {}
    final_mapping = {}
    for q in range(circuit.num_qubits):
        initial_mapping[q] = layouts[0][q]
        final_mapping[q] = layouts[-1][q]
    return MappingResult(circuit=mapped, initial_mapping=initial_mapping, final_mapping=final_mapping)

