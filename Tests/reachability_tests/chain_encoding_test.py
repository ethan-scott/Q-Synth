from qiskit import QuantumCircuit

from Tests.test_utils import EXAMPLES_DIR, get_cx_count
from qsynth.ReachabilitySolver.encodings.chain_synthesis.chain_encoding import chain_synthesis
from qsynth.ReachabilitySolver.encodings.layout_synthesis.layout_reachability_synthesis import \
    get_coupling_graph_from_platform
from qsynth.api import check_equivalence

line3_coupling_graph = [[0, 1], [1, 2]]
tenerife_coupling_graph, tenerife_num_physical_qubits = get_coupling_graph_from_platform("tenerife")


def test_free_layout_avoids_routing():
    # Given a CNOT between the two ends of a 3-qubit line
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)

    # When synthesising with the chain encoding
    result = chain_synthesis(circuit, line3_coupling_graph, 3)

    # Then the layout puts q0 and q2 next to each other: one CNOT
    check_equivalence(circuit, result.circuit, result.final_mapping, result.initial_mapping)
    assert get_cx_count(result.circuit) == 1


def test_two_slices():
    # Given two CNOT slices separated by a T gate on the qubit both of them use
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    circuit.t(2)
    circuit.cx(2, 1)

    # When synthesising with the chain encoding
    result = chain_synthesis(circuit, line3_coupling_graph, 3)

    # Then no routing CNOTs are needed
    check_equivalence(circuit, result.circuit, result.final_mapping, result.initial_mapping)
    assert get_cx_count(result.circuit) == 2


def test_ecai24_on_tenerife():
    # Given the example circuit from the ECAI'24 paper
    circuit = QuantumCircuit.from_qasm_file(f"{EXAMPLES_DIR}/ecai24.qasm")

    # When synthesising with the chain encoding on tenerife
    result = chain_synthesis(circuit, tenerife_coupling_graph, tenerife_num_physical_qubits)

    # Then the solution is correct and optimal
    check_equivalence(circuit, result.circuit, result.final_mapping, result.initial_mapping)
    assert get_cx_count(result.circuit) == 2


def test_sat24_on_tenerife():
    # Given the example circuit from the SAT'24 paper
    circuit = QuantumCircuit.from_qasm_file(f"{EXAMPLES_DIR}/sat24.qasm")

    # When synthesising with the chain encoding on tenerife
    result = chain_synthesis(circuit, tenerife_coupling_graph, tenerife_num_physical_qubits)

    # Then the solution is correct and optimal
    check_equivalence(circuit, result.circuit, result.final_mapping, result.initial_mapping)
    assert get_cx_count(result.circuit) == 6


def test_date24_on_tenerife():
    # Given the example circuit from the DATE'24 paper
    circuit = QuantumCircuit.from_qasm_file(f"{EXAMPLES_DIR}/date24-sub.qasm")

    # When synthesising with the chain encoding on tenerife
    result = chain_synthesis(circuit, tenerife_coupling_graph, tenerife_num_physical_qubits)

    # Then the solution is correct and optimal
    check_equivalence(circuit, result.circuit, result.final_mapping, result.initial_mapping)
    assert get_cx_count(result.circuit) == 5


def test_iccad23_or_on_tenerife():
    # Given the or circuit from the ICCAD'23 benchmarks
    circuit = QuantumCircuit.from_qasm_file("Benchmarks/ICCAD-23/or.qasm")

    # When synthesising with the chain encoding on tenerife
    result = chain_synthesis(circuit, tenerife_coupling_graph, tenerife_num_physical_qubits)

    # Then the solution is correct and optimal
    check_equivalence(circuit, result.circuit, result.final_mapping, result.initial_mapping)
    assert get_cx_count(result.circuit) == 6
