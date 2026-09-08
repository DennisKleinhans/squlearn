"""Tests for transpiler target alias normalization on the PennyLane target.

qulacs no longer has an sQUlearn-side target (util/qulacs/ was removed - qc_executor's
own QulacsExecutor transpiles natively, with its own separate test coverage)."""

from qiskit import QuantumCircuit, transpile

from squlearn.util.pennylane.pennylane_gates import qiskit_pennylane_target


class TestTranspilerTargetAliases:
    """Ensure alias names are normalized to canonical basis-gate names."""

    def test_pennylane_target_normalizes_aliases(self):
        qc = QuantumCircuit(3)
        qc.id(0)
        qc.cx(0, 1)
        qc.ccx(0, 1, 2)

        # Alias names in Qiskit are normalized by instruction names during transpilation.
        transpiled = transpile(qc, target=qiskit_pennylane_target, optimization_level=0)
        op_names = [inst.operation.name for inst in transpiled.data]

        assert all(name in {"id", "cx", "ccx"} for name in op_names)
