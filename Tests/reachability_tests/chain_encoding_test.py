import pytest
from qiskit import QuantumCircuit

from qsynth.ReachabilitySolver.encodings.chain_synthesis.chain_encoding import chain_synthesis
from qsynth.ReachabilitySolver.encodings.layout_synthesis.layout_reachability_synthesis import \
    get_coupling_graph_from_platform
from qsynth.api import check_equivalence

line3 = [(0, 1), (1, 2)]


def test_free_layout_avoids_routing():
    # Given a CNOT between the two ends of a 3-qubit line
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)

    # When synthesising with the chain encoding
    mapped, cost, initial, final = chain_synthesis(circuit, line3, 3)

    # Then the layout puts q0 and q2 next to each other: one CNOT
    assert cost == 1
    check_equivalence(circuit, mapped, final_mapping=final, initial_mapping=initial)


def test_two_sections():
    # Given two CNOT sections separated by a T gate on the qubit both of them use
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    circuit.t(2)
    circuit.cx(2, 1)

    # When synthesising with the chain encoding
    mapped, cost, initial, final = chain_synthesis(circuit, line3, 3)

    # Then the result is equivalent and needs no routing CNOTs
    assert cost == 2
    check_equivalence(circuit, mapped, final_mapping=final, initial_mapping=initial)


@pytest.mark.parametrize("name, optimum", [("Examples/ecai24.qasm", 2), ("Examples/sat24.qasm", 6),
                                           ("Examples/date24-sub.qasm", 5), ("ICCAD-23/or.qasm", 6)])
def test_small_benchmarks_on_tenerife(name, optimum):
    circuit = QuantumCircuit.from_qasm_file(f"Benchmarks/{name}")
    circuit.remove_final_measurements()
    coupling_graph, n = get_coupling_graph_from_platform("tenerife")

    mapped, cost, initial, final = chain_synthesis(circuit, coupling_graph, n)

    assert cost == optimum
    check_equivalence(circuit, mapped, final_mapping=final, initial_mapping=initial)
