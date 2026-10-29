from qiskit import QuantumCircuit

from Tests.test_utils import EXAMPLES_DIR, get_cx_count
from qsynth.ReachabilitySolver.encodings.chain_synthesis.chain_encoding import chain_synthesis
from qsynth.ReachabilitySolver.encodings.layout_synthesis.layout_reachability_synthesis import \
    get_coupling_graph_from_platform
from qsynth.api import check_equivalence

line3_coupling_graph = [[0, 1], [1, 2]]
tenerife_coupling_graph, tenerife_num_physical_qubits = get_coupling_graph_from_platform("tenerife")

# In each test, the input circuit's wires are logical qubits and the result's wires are physical qubits.
# Layouts are written {logical qubit: physical qubit}. Idle wires are not drawn.
# The result is the one the solver returned; other results with the same CNOT count may also be optimal.


def test_free_layout_avoids_routing():
    # Given a CNOT between the two ends of a 3-qubit line
    #
    #   q_0: ──■──
    #        ┌─┴─┐
    #   q_2: ┤ X ├
    #        └───┘
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)

    # When synthesising with the chain encoding
    result = chain_synthesis(circuit, line3_coupling_graph, 3)

    # Then the layout puts q0 and q2 next to each other: one CNOT
    #
    #   initial layout {0: 1, 1: 0, 2: 2}, final layout {0: 1, 1: 0, 2: 2}
    #   q_1: ──■──
    #        ┌─┴─┐
    #   q_2: ┤ X ├
    #        └───┘
    check_equivalence(circuit, result.circuit, result.final_mapping, result.initial_mapping)
    assert get_cx_count(result.circuit) == 1


def test_two_slices():
    # Given two CNOT slices separated by a T gate on the qubit both of them use
    #
    #   q_0: ──■────────────
    #          │       ┌───┐
    #   q_1: ──┼───────┤ X ├
    #        ┌─┴─┐┌───┐└─┬─┘
    #   q_2: ┤ X ├┤ T ├──■──
    #        └───┘└───┘
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    circuit.t(2)
    circuit.cx(2, 1)

    # When synthesising with the chain encoding
    result = chain_synthesis(circuit, line3_coupling_graph, 3)

    # Then no routing CNOTs are needed: q2 is placed in the middle of the line
    #
    #   initial layout {0: 2, 1: 0, 2: 1}, final layout {0: 2, 1: 0, 2: 1}
    #                  ┌───┐
    #   q_0: ──────────┤ X ├
    #        ┌───┐┌───┐└─┬─┘
    #   q_1: ┤ X ├┤ T ├──■──
    #        └─┬─┘└───┘
    #   q_2: ──■────────────
    check_equivalence(circuit, result.circuit, result.final_mapping, result.initial_mapping)
    assert get_cx_count(result.circuit) == 2


def test_ecai24_on_tenerife():
    # Given the example circuit from the ECAI'24 paper
    #
    #                  ┌───┐     ┌───┐
    #   q_0: ──────────┤ X ├─────┤ X ├─────
    #        ┌───┐     └─┬─┘┌───┐└─┬─┘┌───┐
    #   q_1: ┤ X ├──■────■──┤ X ├──■──┤ X ├
    #        └─┬─┘┌─┴─┐     └─┬─┘     └─┬─┘
    #   q_3: ──■──┤ X ├───────■─────────■──
    #             └───┘
    circuit = QuantumCircuit.from_qasm_file(f"{EXAMPLES_DIR}/ecai24.qasm")

    # When synthesising with the chain encoding on tenerife
    result = chain_synthesis(circuit, tenerife_coupling_graph, tenerife_num_physical_qubits)

    # Then the solution is correct and optimal: 6 CNOTs become 2, the rest is absorbed by the final layout
    #
    #   initial layout {0: 1, 1: 0, 2: 4, 3: 2}, final layout {0: 1, 1: 2, 2: 4, 3: 0}
    #   q_0: ──■────■──
    #          │  ┌─┴─┐
    #   q_1: ──┼──┤ X ├
    #        ┌─┴─┐└───┘
    #   q_2: ┤ X ├─────
    #        └───┘
    check_equivalence(circuit, result.circuit, result.final_mapping, result.initial_mapping)
    assert get_cx_count(result.circuit) == 2


def test_sat24_on_tenerife():
    # Given the example circuit from the SAT'24 paper
    #
    #         ┌───┐ ┌─────┐              ┌───┐┌───┐            ┌───┐
    #   q_0: ─┤ X ├─┤ Tdg ├──────────■───┤ X ├┤ T ├────────────┤ X ├──■──
    #        ┌┴───┴┐└┬───┬┘ ┌───┐  ┌─┴─┐ └─┬─┘├───┤┌───┐┌─────┐└─┬─┘┌─┴─┐
    #   q_1: ┤ Tdg ├─┤ X ├──┤ T ├──┤ X ├───┼──┤ T ├┤ X ├┤ Tdg ├──┼──┤ X ├
    #        └┬───┬┘ └─┬─┘ ┌┴───┴┐┌┴───┴┐  │  └───┘└─┬─┘└─────┘  │  ├───┤
    #   q_2: ─┤ H ├────■───┤ Tdg ├┤ Sdg ├──■─────────■───────────■──┤ H ├
    #         └───┘        └─────┘└─────┘                           └───┘
    circuit = QuantumCircuit.from_qasm_file(f"{EXAMPLES_DIR}/sat24.qasm")

    # When synthesising with the chain encoding on tenerife
    result = chain_synthesis(circuit, tenerife_coupling_graph, tenerife_num_physical_qubits)

    # Then the solution is correct and optimal: all three logical qubits are placed on a triangle, so no routing
    #
    #   initial layout {0: 2, 1: 0, 2: 1}, final layout {0: 2, 1: 0, 2: 1}
    #        ┌─────┐ ┌───┐  ┌───┐        ┌───┐┌───┐          ┌───┐┌─────┐┌───┐
    #   q_0: ┤ Tdg ├─┤ X ├──┤ T ├────────┤ X ├┤ T ├──────────┤ X ├┤ Tdg ├┤ X ├
    #        └┬───┬┘ └─┬─┘ ┌┴───┴┐┌─────┐└─┬─┘└───┘          └─┬─┘└┬───┬┘└─┬─┘
    #   q_1: ─┤ H ├────■───┤ Tdg ├┤ Sdg ├──┼────■─────────■────■───┤ H ├───┼──
    #         ├───┤ ┌─────┐└─────┘└─────┘  │  ┌─┴─┐┌───┐┌─┴─┐      └───┘   │
    #   q_2: ─┤ X ├─┤ Tdg ├────────────────■──┤ X ├┤ T ├┤ X ├──────────────■──
    #         └───┘ └─────┘                   └───┘└───┘└───┘
    check_equivalence(circuit, result.circuit, result.final_mapping, result.initial_mapping)
    assert get_cx_count(result.circuit) == 6


def test_date24_on_tenerife():
    # Given the example circuit from the DATE'24 paper: a ring of 4 CNOTs
    #
    #                       ┌───┐
    #   q_0: ──■────────────┤ X ├
    #        ┌─┴─┐          └─┬─┘
    #   q_1: ┤ X ├──■─────────┼──
    #        └───┘┌─┴─┐       │
    #   q_2: ─────┤ X ├──■────┼──
    #             └───┘┌─┴─┐  │
    #   q_3: ──────────┤ X ├──■──
    #                  └───┘
    circuit = QuantumCircuit.from_qasm_file(f"{EXAMPLES_DIR}/date24-sub.qasm")

    # When synthesising with the chain encoding on tenerife
    result = chain_synthesis(circuit, tenerife_coupling_graph, tenerife_num_physical_qubits)

    # Then the solution is correct and optimal: one extra CNOT, and q2 and q3 swap places in the final layout
    #
    #   initial layout {0: 0, 1: 1, 2: 2, 3: 3}, final layout {0: 0, 1: 1, 2: 3, 3: 2}
    #                       ┌───┐
    #   q_0: ──■────────────┤ X ├─────
    #        ┌─┴─┐          └─┬─┘
    #   q_1: ┤ X ├──■─────────┼───────
    #        └───┘┌─┴─┐┌───┐  │
    #   q_2: ─────┤ X ├┤ X ├──■────■──
    #             └───┘└─┬─┘     ┌─┴─┐
    #   q_3: ────────────■───────┤ X ├
    #                            └───┘
    check_equivalence(circuit, result.circuit, result.final_mapping, result.initial_mapping)
    assert get_cx_count(result.circuit) == 5


# CNOT counts on tenerife, compared with earlier pipelines (work/chain/compare_tenerife.log, 2026-09-27)
#   input: CNOT count of the original circuit (ignores the coupling graph, so not a fair comparison)
#   A:     1. layout_synthesis (SAT'24): choose an initial layout and insert the fewest SWAPs
#          2. cnot_peephole_synthesis (ECAI'24) with the coupling graph: re-synthesise each CNOT slice,
#             keeping the layout from step 1 fixed (S+R)
#   B:     1. cnot_peephole_synthesis without a coupling graph, output permutation allowed (W)
#          2. and 3. same as A
#   Cliff: 1. layout_synthesis, as in A
#          2. clifford_peephole_synthesis with the coupling graph: re-synthesise Clifford slices instead of CNOT slices
#   chain: this encoding (layouts and CNOTs of all slices solved together)
#
#   circuit       input   A   B   Cliff   chain
#   ecai24          6     3   2     3       2
#   sat24           6     6   6     6       6
#   date24-sub      4     7   7     6       5
