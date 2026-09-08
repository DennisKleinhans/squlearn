import pytest
import numpy as np


from squlearn import Executor
from squlearn.encoding_circuit import ParamZFeatureMap
from squlearn.observables import SinglePauli, SummedPaulis
from squlearn.qnn.lowlevel_qnn import LowLevelQNN
from squlearn.qnn.lowlevel_qnn.lowlevel_qnn_qiskit import LowLevelQNNQiskit
from squlearn.qnn.lowlevel_qnn.lowlevel_qnn_pennylane import LowLevelQNNPennyLane


def get_values(framework):
    executor = Executor(framework)
    pqc = ParamZFeatureMap(4, 2)
    obs1 = SummedPaulis(4)
    obs2 = SummedPaulis(4)

    llqnn = LowLevelQNN(pqc, [obs1, obs2], executor=executor, num_features=2)

    np.random.seed(42)
    param = np.random.rand(2, llqnn.num_parameters)
    param_pbs = np.random.rand(2, llqnn.num_parameters_observable)

    # Each framework's llqnn owns its own Parameters vector (e.g. "p" for qiskit, "param"
    # for pennylane), so the tuple key for "gradient w.r.t. the first PQC/observable
    # parameter" is a different object per framework - keep it alongside the result dict
    # instead of trying to relocate it positionally from the far side of the call.
    param_key = (llqnn.parameters[0],)
    param_op_key = (llqnn.parameters_operator[0],)

    values = llqnn.evaluate(
        [[0.1, 0.2], [0.3, 0.4]],
        param,
        param_pbs,
        "f",
        "dfdp",
        "dfdx",
        "var",
        param_key,
        param_op_key,
    )
    return values, param_key, param_op_key


def test_backends_consistency():
    """Tests that different derivatives computed with different frameworks are consistent.

    Qulacs is intentionally excluded here: this test relies on tuple-based
    derivative specs built from llqnn.parameters[i]/parameters_operator[i],
    which - like every non-native derivative key - route through the
    framework-specific fallback engine. Qulacs has none (see
    LowLevelQNNUnified._fallback); its native-key agreement with the other
    two frameworks is covered separately, see
    test_var_family_is_native_and_matches_legacy_engine's git history for the
    bit-exact proof recorded before the legacy Qulacs engine was removed, and
    test_multiple_output_shape_with_n_observables below for its continued
    coverage of the still-supported native keys.
    """

    values_qiskit, qiskit_param_key, qiskit_param_op_key = get_values("qiskit")
    values_pennylane, pennylane_param_key, pennylane_param_op_key = get_values("pennylane")

    for k in ["f", "dfdp", "dfdx", "var"]:
        assert np.allclose(values_qiskit[k], values_pennylane[k])

    assert np.allclose(values_qiskit[qiskit_param_key], values_pennylane[pennylane_param_key])

    assert np.allclose(
        values_qiskit[qiskit_param_op_key], values_pennylane[pennylane_param_op_key]
    )


@pytest.mark.parametrize("framework", ["pennylane", "qiskit", "qulacs"])
@pytest.mark.parametrize("n_obs", [1, 2])
def test_multiple_output_shape_with_n_observables(framework, n_obs):
    """Regression: a list-of-one observable must produce the same shape contract
    as a list of multiple observables. PennyLane otherwise collapses the leading
    "observable" axis when only one measurement is returned, which previously
    crashed ProjectedQuantumKernel for num_qubits=1 with a single-Pauli measurement.
    """
    pqc = ParamZFeatureMap(4, 2)
    obs = [SinglePauli(4, i, op_str="X") for i in range(n_obs)]
    llqnn = LowLevelQNN(pqc, obs, executor=Executor(framework), num_features=2)

    np.random.seed(42)
    x = np.random.rand(5, 2)
    param = np.random.rand(llqnn.num_parameters)

    assert llqnn.evaluate(x, param, [], "f")["f"].shape == (5, n_obs)
    assert llqnn.evaluate(x, param, [], "dfdx")["dfdx"].shape == (5, n_obs, 2)
    assert llqnn.evaluate(x, param, [], "dfdp")["dfdp"].shape == (
        5,
        n_obs,
        llqnn.num_parameters,
    )


_LEGACY_ENGINE = {
    "qiskit": LowLevelQNNQiskit,
    "pennylane": LowLevelQNNPennyLane,
}

_VAR_FAMILY_KEYS = ("var", "varf", "dvardx", "dvardp", "dvardop")


@pytest.mark.parametrize("framework", ["qiskit", "pennylane"])
@pytest.mark.parametrize(
    "observable",
    [
        pytest.param(lambda pqc: SummedPaulis(pqc.num_qubits), id="single-parameterized"),
        pytest.param(
            lambda pqc: [SinglePauli(pqc.num_qubits, i, op_str="Z") for i in range(3)],
            id="multi-parameter-free",
        ),
        pytest.param(
            lambda pqc: [
                SinglePauli(pqc.num_qubits, 0, op_str="Z"),
                SummedPaulis(pqc.num_qubits),
            ],
            id="multi-mixed-parameters",
        ),
    ],
)
def test_var_family_is_native_and_matches_legacy_engine(framework, observable):
    """The var/dvardx/dvardp/dvardop family is evaluated via qc_executor's native
    <O^2> path (LowLevelQNNUnified._native_observable_squared), not the legacy
    per-framework fallback engine - verified both by the fallback never being built
    and by bit-for-bit agreement with the legacy engine's own computation.

    Qulacs is not parametrized here: it has no legacy engine left to compare
    against (removed together with LowLevelQNNQulacs/util/qulacs/), so it is
    covered separately by test_var_family_is_native_for_qulacs below. Before
    the removal, this same test - then parametrized over qiskit/pennylane/
    qulacs alike - passed for all three, which is the bit-exact proof that
    the removal was safe for the native key set.
    """
    pqc = ParamZFeatureMap(3, 2)
    obs = observable(pqc)
    rng = np.random.default_rng(3)
    x = rng.random((3, 2))
    param = rng.random(pqc.num_parameters)
    num_parameters_observable = (
        sum(o.num_parameters for o in obs) if isinstance(obs, list) else obs.num_parameters
    )
    param_op = rng.random(num_parameters_observable)

    executor = Executor("statevector_simulator") if framework == "qiskit" else Executor(framework)
    llqnn = LowLevelQNN(pqc, obs, executor, num_features=2)
    native = llqnn.evaluate(x, param, param_op, "f", *_VAR_FAMILY_KEYS)
    assert llqnn._fallback_engine is None

    # dvardop has a zero-sized trailing axis when no observable owns any "p_op"
    # parameter at all - LowLevelQNNPennyLane's legacy engine crashes on that shape
    # (pre-existing, unrelated to this native path), so skip the legacy comparison
    # for exactly that combination while still exercising it on the native path above.
    comparison_keys = ("f", *_VAR_FAMILY_KEYS)
    if num_parameters_observable == 0:
        comparison_keys = tuple(k for k in comparison_keys if k != "dvardop")

    legacy_executor = (
        Executor("statevector_simulator") if framework == "qiskit" else Executor(framework)
    )
    legacy = _LEGACY_ENGINE[framework](pqc, obs, legacy_executor, num_features=2).evaluate(
        x, param, param_op, *comparison_keys
    )

    for key in comparison_keys:
        np.testing.assert_allclose(native[key], legacy[key], atol=1e-8)


def test_var_family_is_native_for_qulacs():
    """Qulacs has no legacy engine to compare against (see the test above), so
    this checks the same native <O^2> var-family path against a qiskit
    statevector reference instead - real cross-framework agreement, not a
    self-consistency check against qc_executor's own computation."""
    pqc = ParamZFeatureMap(3, 2)
    obs = SummedPaulis(pqc.num_qubits)
    rng = np.random.default_rng(3)
    x = rng.random((3, 2))
    param = rng.random(pqc.num_parameters)
    param_op = rng.random(obs.num_parameters)

    llqnn_qulacs = LowLevelQNN(pqc, obs, Executor("qulacs"), num_features=2)
    native = llqnn_qulacs.evaluate(x, param, param_op, "f", *_VAR_FAMILY_KEYS)
    assert llqnn_qulacs._fallback_engine is None

    llqnn_qiskit = LowLevelQNN(pqc, obs, Executor("statevector_simulator"), num_features=2)
    reference = llqnn_qiskit.evaluate(x, param, param_op, "f", *_VAR_FAMILY_KEYS)

    for key in ("f", *_VAR_FAMILY_KEYS):
        np.testing.assert_allclose(native[key], reference[key], atol=1e-8)
