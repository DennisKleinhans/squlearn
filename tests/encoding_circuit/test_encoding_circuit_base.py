from math import inf
import pytest
import warnings
import numpy as np
from qiskit import QuantumCircuit
from squlearn.encoding_circuit.encoding_circuit_base import EncodingCircuitBase


class MockCircuitBase(EncodingCircuitBase):
    def get_circuit(self, features, parameters):
        return QuantumCircuit(self.num_qubits)

    @property
    def num_parameters(self):
        return 2

    @property
    def num_encoding_slots(self):
        return 10


class TestEncodingCircuitBase:

    def test_init(self):
        circuit = MockCircuitBase(num_features=2, num_qubits=2)
        assert circuit.num_features == 2
        assert circuit.num_qubits == 2

    def test_generate_initial_parameters(self):
        custom_circuit = MockCircuitBase(num_qubits=4)
        params = custom_circuit.generate_initial_parameters(seed=42)
        assert len(params) == 2
        assert (params >= -np.pi).all() and (params <= np.pi).all()

    def test_draw_warning(self):
        circuit = MockCircuitBase(num_qubits=4)
        with warnings.catch_warnings(record=True) as w:
            circuit.draw()
            assert len(w) == 1
            assert "`num_features` is not set" in str(w[-1].message)

    def test_add(self):
        circuit_1 = MockCircuitBase(num_qubits=4, num_features=2)
        circuit_2 = MockCircuitBase(num_qubits=4, num_features=3)
        circuit_3 = MockCircuitBase(num_qubits=3, num_features=2)
        circuit_composed = circuit_1 + circuit_2

        # check if the composed circuit has the correct number of qubits, features, and parameters
        assert circuit_composed.num_qubits == 4
        assert circuit_composed.num_features == 3
        assert circuit_composed.num_parameters == 4

        # check if the composed circuit has the correct number of parameters
        composed_params = circuit_composed.generate_initial_parameters(seed=42)
        assert len(composed_params) == 4
        assert (composed_params >= -np.pi).all() and (composed_params <= np.pi).all()

        # check if the composed circuit has the correct named parameters
        composed_named_params = circuit_composed.get_params()
        assert composed_named_params == {
            "ec1": circuit_1,
            "ec2": circuit_2,
            "ec1__num_features": 2,
            "ec2__num_features": 3,
            "num_qubits": 4,
        }

        # check if the composed circuit can be set with new parameters
        circuit_composed.set_params(num_qubits=5, ec1__num_features=3, ec2__num_features=4)
        assert circuit_composed.num_qubits == 5
        assert circuit_1.num_features == 3
        assert circuit_2.num_features == 4

        # unequal number of qubits
        with pytest.raises(ValueError):
            circuit_1 + circuit_3

        # invalid type
        with pytest.raises(ValueError):
            circuit_1 + "invalid"

    def test_get_and_set_params(self):
        circuit = MockCircuitBase(num_qubits=4, num_features=2)
        params = circuit.get_params()
        assert params == {"num_qubits": 4, "num_features": 2}

        circuit.set_params(num_qubits=5, num_features=3)
        assert circuit.num_qubits == 5
        assert circuit.num_features == 3

        with pytest.raises(ValueError):
            circuit.set_params(invalid_param=1)