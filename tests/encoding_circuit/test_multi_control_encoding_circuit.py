import numpy as np
import pytest
from qiskit import QuantumCircuit
from squlearn.encoding_circuit import MultiControlEncodingCircuit
from squlearn.encoding_circuit.encoding_circuit_base import EncodingSlotsMismatchError


class TestMultiControlEncodingCircuit:

    def test_init(self):
        circuit = MultiControlEncodingCircuit(num_qubits=2, num_features=2)
        assert circuit.num_features == 2
        assert circuit.num_qubits == 2
        assert circuit.num_layers == 1
        assert circuit.closed is True
        assert circuit.final_encoding is False

        with pytest.raises(ValueError):
            MultiControlEncodingCircuit(num_qubits=1, num_features=2)

    def test_num_parameters_closed(self):
        features = 2
        qubits = 3
        layers = 1
        circuit = MultiControlEncodingCircuit(
            num_features=features,
            num_qubits=qubits,
            num_layers=layers,
            closed=True,
        )
        assert circuit.num_parameters == 3 * (qubits - 1) * layers + 3 * layers

    def test_num_parameters_none_closed(self):
        features = 2
        qubits = 3
        layers = 1
        circuit = MultiControlEncodingCircuit(
            num_features=features,
            num_qubits=qubits,
            num_layers=layers,
            closed=False,
        )
        assert circuit.num_parameters == 3 * (qubits - 1) * layers

    def test_parameter_bounds(self):
        qubits = 3
        circuit = MultiControlEncodingCircuit(num_qubits=qubits, num_layers=1)
        bounds = circuit.parameter_bounds
        assert bounds.shape == (circuit.num_parameters, 2)
        assert np.all(bounds == [-2.0 * np.pi, 2.0 * np.pi])

    def test_get_params(self):
        circuit = MultiControlEncodingCircuit(num_features=2, num_qubits=2)
        named_params = circuit.get_params()
        assert named_params == {
            "num_features": 2,
            "num_qubits": 2,
            "num_layers": 1,
            "closed": True,
            "final_encoding": False,
        }

    def test_get_circuit(self):
        circuit = MultiControlEncodingCircuit(num_features=2, num_qubits=2)
        features = np.array([0.5, -0.5])
        params = np.random.uniform(-2.0 * np.pi, 2.0 * np.pi, circuit.num_parameters)
        qc = circuit.get_circuit(features=features, parameters=params)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

        with pytest.raises(EncodingSlotsMismatchError):
            MultiControlEncodingCircuit(num_qubits=2, num_features=1).get_circuit(
                features=[0.3, 0.5, -0.5], parameters=params
            )