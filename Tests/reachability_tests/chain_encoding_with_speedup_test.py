import pytest
from qiskit import QuantumCircuit

from Tests.test_utils import get_cx_count
from qsynth.ReachabilitySolver.encodings.chain_synthesis import chain_encoding, chain_encoding_with_speedup
from qsynth.ReachabilitySolver.encodings.layout_synthesis.layout_reachability_synthesis import \
    get_coupling_graph_from_platform
from qsynth.api import check_equivalence

# (platform, circuit, optimal CNOT count)
SMALL = [("tenerife", "Examples/ecai24.qasm", 2), ("tenerife", "Examples/sat24.qasm", 6),
         ("tenerife", "Examples/date24-sub.qasm", 5)]
LARGER = [("tenerife", "Examples/adder_tgate_optimal.qasm", 9), ("tenerife", "ICCAD-23/qaoa5.qasm", 8),
          ("tenerife", "Examples/iccad23.qasm", 12), ("tenerife", "ICCAD-23/adder.qasm", 12),
          ("tenerife", "ICCAD-23/4mod5-v1_22.qasm", 12),
          ("melbourne", "Random-Clifford/tket_optimized_without_swaps_no_u3_gates/04q_33936.qasm", 11),
          ("melbourne", "Feynman/tket_optimized_without_swaps/tof_3.qasm", 24),
          ("melbourne", "ICCAD-23/mod5mils_65.qasm", 24),
          ("melbourne", "Random-Clifford/tket_optimized_without_swaps_no_u3_gates/05q_05306.qasm", 17)]


@pytest.mark.parametrize("platform, name, optimum", SMALL)
def test_same_optimum_as_reference(platform, name, optimum):
    # Given a small benchmark circuit
    circuit = QuantumCircuit.from_qasm_file(f"Benchmarks/{name}")
    coupling_graph, num_physical_qubits = get_coupling_graph_from_platform(platform)

    # When synthesising with and without the speedup
    reference = chain_encoding.chain_synthesis(circuit, coupling_graph, num_physical_qubits)
    result = chain_encoding_with_speedup.chain_synthesis(circuit, coupling_graph, num_physical_qubits)

    # Then both find the optimum
    assert get_cx_count(reference.circuit) == optimum
    assert get_cx_count(result.circuit) == optimum


@pytest.mark.parametrize("platform, name, optimum", SMALL + LARGER)
def test_benchmarks(platform, name, optimum):
    # Given a benchmark circuit
    circuit = QuantumCircuit.from_qasm_file(f"Benchmarks/{name}")
    coupling_graph, num_physical_qubits = get_coupling_graph_from_platform(platform)

    # When synthesising with the speedup
    result = chain_encoding_with_speedup.chain_synthesis(circuit, coupling_graph, num_physical_qubits)

    # Then the solution is correct and optimal
    check_equivalence(circuit, result.circuit, result.final_mapping, result.initial_mapping)
    assert get_cx_count(result.circuit) == optimum
