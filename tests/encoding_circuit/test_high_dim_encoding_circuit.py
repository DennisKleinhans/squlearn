import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from squlearn.encoding_circuit import HighDimEncodingCircuit


class TestHighDimEncodingCircuit:

    def test_init(self):
        circuit = HighDimEncodingCircuit(num_qubits=2, num_features=2)
        assert circuit.num_features == 2
        assert circuit.num_qubits == 2
        assert circuit.cycling is True
        assert circuit.cycling_type == "saw"
        assert circuit.num_layers is None
        assert circuit.layer_type == "rows"
        assert circuit.entangling_gate == "iswap"

        with pytest.raises(ValueError):
            HighDimEncodingCircuit(num_features=2, num_qubits=2, cycling_type="invalid")

        with pytest.raises(ValueError):
            HighDimEncodingCircuit(num_features=2, num_qubits=2, layer_type="invalid")

        with pytest.raises(ValueError):
            HighDimEncodingCircuit(num_features=2, num_qubits=2, entangling_gate="invalid")

    def test_get_params(self):
        circuit = HighDimEncodingCircuit(num_features=2, num_qubits=2)
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": 2,
            "num_qubits": 2,
            "num_layers": None,
            "cycling": True,
            "cycling_type": "saw",
            "layer_type": "rows",
            "entangling_gate": "iswap",
        }

    def test_get_circuit(self):
        circuit = HighDimEncodingCircuit(num_features=2, num_qubits=2)
        features = np.array([0.5, -0.5])

        qc = circuit.get_circuit(features=features)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

        circuit = HighDimEncodingCircuit(num_qubits=4, num_features=12)
        features = ParameterVector("x", 12)
        qc = circuit.get_circuit(features)
        assert circuit.num_layers == 2
