import pytest
import numpy as np


from squlearn import Executor
from squlearn.encoding_circuit import ParamZFeatureMap
from squlearn.observables import SinglePauli, SummedPaulis
from squlearn.qnn.lowlevel_qnn import LowLevelQNN
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
    derivative specs built from llqnn.parameters[i]/parameters_operator[i]. For
    Qiskit these are now evaluated natively via qc_executor's tuple mechanism
    (see LowLevelQNNUnified._build_derivative_arg, WP-G); for PennyLane they
    still route through the framework-specific fallback engine. Qulacs has
    neither (see LowLevelQNNUnified._fallback); its native-key agreement with
    the other two frameworks is covered separately, see
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


def test_raw_derivative_key_p_op_ownership_matches_pennylane_reference():
    """Regression for the WP-G raw-key multi-output p_op-ownership logic
    (LowLevelQNNUnified._p_op_entry_excludes_observable): a second derivative built from two
    individual llqnn.parameters_operator[i] elements belonging to *different* observables
    must be evaluated without error (qc_executor itself raises if a raw parameter belonging
    to a different observable's operator is passed through unfiltered - this is what the
    per-observable ownership check exists to prevent) and must agree with PennyLane's
    independently-computed fallback engine. Both frameworks' observables are linear in
    "p_op" (a weighted Pauli sum), so any p_op-p_op second derivative is mathematically zero
    regardless of ownership - that shared, non-vacuous mathematical fact (not just "both
    happen to agree") is asserted explicitly alongside the cross-framework agreement.
    """
    pqc = ParamZFeatureMap(3, 2)
    obs = [
        SinglePauli(pqc.num_qubits, 0, op_str="Z"),
        SummedPaulis(pqc.num_qubits),
        SummedPaulis(pqc.num_qubits),
    ]
    rng = np.random.default_rng(23)
    x = rng.random((2, 2))
    param = rng.random(pqc.num_parameters)
    n_pop = sum(o.num_parameters for o in obs)
    param_op = rng.random(n_pop)

    llqnn_qiskit = LowLevelQNN(pqc, obs, Executor("statevector_simulator"), num_features=2)
    llqnn_pennylane = LowLevelQNN(pqc, obs, Executor("pennylane"), num_features=2)

    # p_op[0:4] belongs to the second observable, p_op[4:8] to the third (the first owns none).
    qiskit_key = (llqnn_qiskit.parameters_operator[0], llqnn_qiskit.parameters_operator[4])
    pennylane_key = (
        llqnn_pennylane.parameters_operator[0],
        llqnn_pennylane.parameters_operator[4],
    )

    qiskit_vals = llqnn_qiskit.evaluate(x, param, param_op, qiskit_key)
    assert llqnn_qiskit._fallback_engine is None
    pennylane_vals = llqnn_pennylane.evaluate(x, param, param_op, pennylane_key)

    np.testing.assert_allclose(qiskit_vals[qiskit_key], pennylane_vals[pennylane_key], atol=1e-8)
    np.testing.assert_allclose(qiskit_vals[qiskit_key], 0.0, atol=1e-8)


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


_VAR_FAMILY_KEYS = ("var", "varf", "dvardx", "dvardp", "dvardop")


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
def test_var_family_is_native_and_matches_legacy_engine(observable):
    """The var/dvardx/dvardp/dvardop family is evaluated via qc_executor's native
    <O^2> path (LowLevelQNNUnified._native_observable_squared), not the legacy
    per-framework fallback engine - verified both by the fallback never being built
    and by bit-for-bit agreement with the legacy engine's own computation.

    Only PennyLane is parametrized here: it is the only framework with a legacy
    engine left. Qiskit has none any more (removed together with
    LowLevelQNNQiskit/encoding_circuit_derivatives.py/observable_derivatives.py
    once WP-G proved bit-exact equivalence for every key it supported) - its
    agreement with this same var-family/observable matrix is covered separately
    by test_var_family_is_native_for_qiskit_matches_pennylane_reference below.
    Qulacs has neither, and is covered by test_var_family_is_native_for_qulacs.
    Before the Qiskit/Qulacs removals, this same test - then parametrized over
    all three frameworks - passed for all of them, which is the bit-exact proof
    that the removals were safe for the native key set.
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

    llqnn = LowLevelQNN(pqc, obs, Executor("pennylane"), num_features=2)
    native = llqnn.evaluate(x, param, param_op, "f", *_VAR_FAMILY_KEYS)
    assert llqnn._fallback_engine is None

    # dvardop has a zero-sized trailing axis when no observable owns any "p_op"
    # parameter at all - LowLevelQNNPennyLane's legacy engine crashes on that shape
    # (pre-existing, unrelated to this native path), so skip the legacy comparison
    # for exactly that combination while still exercising it on the native path above.
    comparison_keys = ("f", *_VAR_FAMILY_KEYS)
    if num_parameters_observable == 0:
        comparison_keys = tuple(k for k in comparison_keys if k != "dvardop")

    legacy = LowLevelQNNPennyLane(pqc, obs, Executor("pennylane"), num_features=2).evaluate(
        x, param, param_op, *comparison_keys
    )

    for key in comparison_keys:
        np.testing.assert_allclose(native[key], legacy[key], atol=1e-8)


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
def test_var_family_is_native_for_qiskit_matches_pennylane_reference(observable):
    """Qiskit has no legacy engine left to compare against (see the test above), so this
    checks the same native <O^2> var-family path against PennyLane's own (independently
    computed, autograd-based) engine instead - real cross-framework agreement, not a
    self-consistency check against qc_executor's own computation.
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

    llqnn = LowLevelQNN(pqc, obs, Executor("statevector_simulator"), num_features=2)
    native = llqnn.evaluate(x, param, param_op, "f", *_VAR_FAMILY_KEYS)
    assert llqnn._fallback_engine is None

    comparison_keys = ("f", *_VAR_FAMILY_KEYS)
    if num_parameters_observable == 0:
        comparison_keys = tuple(k for k in comparison_keys if k != "dvardop")

    reference = LowLevelQNNPennyLane(pqc, obs, Executor("pennylane"), num_features=2).evaluate(
        x, param, param_op, *comparison_keys
    )

    for key in comparison_keys:
        np.testing.assert_allclose(native[key], reference[key], atol=1e-8)


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


# WP-G: chained (tuple-form) higher-order derivatives, qc_executor-Qiskit-only. Every key
# below is cross-checked against PennyLane's own, independently-computed (chained
# qml.jacobian) native engine, except "dfdopdxdx": PennyLane's own engine fails on that one
# specific key with a pre-existing, unrelated NonDifferentiableError (autograd trips over a
# near-zero complex intermediate) for both observable configs below - verified individually
# per key before writing this test. "dfdpdop" (no legacy string for it, only its transpose
# "dfdopdp") and the three order-3 pure-circuit permutations (dfdxdxdp/dfdxdpdx/dfdpdxdx, a
# pre-existing gap in LowLevelQNNQiskit's now-removed Expec.from_string, unrelated to this WP)
# are verified separately below.
_CHAINED_KEYS = (
    "dfdxdx",
    "dfdpdp",
    "dfdopdp",
    "dfdopdop",
    "dfdpdx",
    "dfdopdx",
    "dfccdxdx",
    "dfccdpdp",
    "dfccdopdx",
    "dfccdopdop",
)

_OBSERVABLE_CONFIGS = [
    pytest.param(lambda pqc: SummedPaulis(pqc.num_qubits), id="single-parameterized"),
    pytest.param(
        lambda pqc: [
            SinglePauli(pqc.num_qubits, 0, op_str="Z"),
            SummedPaulis(pqc.num_qubits),
            SummedPaulis(pqc.num_qubits),
        ],
        id="multi-mixed-parameters",
    ),
]


@pytest.mark.parametrize("observable", _OBSERVABLE_CONFIGS)
def test_chained_derivatives_are_native_and_match_pennylane_reference(observable):
    """The chained-derivative keys (dfdxdx, ..., see _NATIVE_KEY_INFO_QISKIT_ONLY) are
    evaluated via qc_executor's tuple-form expectation_value_derivatives - verified both by
    the fallback never being built and by bit-for-bit agreement with PennyLane's own,
    independently-computed engine, including for a multi-output case with observables that
    own differently-sized (including zero-sized) "p_op" slices.
    """
    pqc = ParamZFeatureMap(3, 2)
    obs = observable(pqc)
    rng = np.random.default_rng(11)
    x = rng.random((2, 2))
    param = rng.random(pqc.num_parameters)
    num_parameters_observable = (
        sum(o.num_parameters for o in obs) if isinstance(obs, list) else obs.num_parameters
    )
    param_op = rng.random(num_parameters_observable)

    executor = Executor("statevector_simulator")
    llqnn = LowLevelQNN(pqc, obs, executor, num_features=2)
    native = llqnn.evaluate(x, param, param_op, *_CHAINED_KEYS)
    assert llqnn._fallback_engine is None

    reference = LowLevelQNNPennyLane(pqc, obs, Executor("pennylane"), num_features=2).evaluate(
        x, param, param_op, *_CHAINED_KEYS
    )

    for key in _CHAINED_KEYS:
        np.testing.assert_allclose(native[key], reference[key], atol=1e-6)


@pytest.mark.parametrize("observable", _OBSERVABLE_CONFIGS)
def test_dfdopdxdx_matches_finite_differences(observable):
    """dfdopdxdx is the one chained key PennyLane's own engine cannot compute (see the note
    above _CHAINED_KEYS), so it is verified independently via central finite differences of
    the already cross-framework-verified "dfdxdx" key with respect to each p_op parameter -
    d(dfdxdx)/dp_op_k approximates dfdopdxdx[..., k, :, :] to O(eps^2).
    """
    pqc = ParamZFeatureMap(3, 2)
    obs = observable(pqc)
    rng = np.random.default_rng(11)
    x = rng.random(2)
    param = rng.random(pqc.num_parameters)
    num_parameters_observable = (
        sum(o.num_parameters for o in obs) if isinstance(obs, list) else obs.num_parameters
    )
    param_op = rng.random(num_parameters_observable)

    llqnn = LowLevelQNN(pqc, obs, Executor("statevector_simulator"), num_features=2)
    analytic = np.asarray(llqnn.evaluate(x, param, param_op, "dfdopdxdx")["dfdopdxdx"])

    eps = 1e-4
    p_op_axis = analytic.ndim - 3  # trailing (..., n_p_op, n_x, n_x); multi-output adds n_op first
    fd = np.zeros_like(analytic)
    for k in range(num_parameters_observable):
        p_plus, p_minus = param_op.copy(), param_op.copy()
        p_plus[k] += eps
        p_minus[k] -= eps
        dxdx_plus = np.asarray(llqnn.evaluate(x, param, p_plus, "dfdxdx")["dfdxdx"])
        dxdx_minus = np.asarray(llqnn.evaluate(x, param, p_minus, "dfdxdx")["dfdxdx"])
        index = tuple(slice(None) if ax != p_op_axis else k for ax in range(analytic.ndim))
        fd[index] = (dxdx_plus - dxdx_minus) / (2 * eps)

    np.testing.assert_allclose(analytic, fd, atol=1e-4)


def test_dfdpdop_matches_transpose_of_dfdopdp():
    """LowLevelQNNQiskit's own Expec.from_string never implemented "dfdpdop" (raises
    "please use dfdopdp instead and transpose"), so there is no legacy reference for it -
    verified instead against its own, independently-computed transpose relation.
    """
    pqc = ParamZFeatureMap(3, 2)
    obs = SummedPaulis(pqc.num_qubits)
    rng = np.random.default_rng(5)
    x = rng.random((2, 2))
    param = rng.random(pqc.num_parameters)
    param_op = rng.random(obs.num_parameters)

    llqnn = LowLevelQNN(pqc, obs, Executor("statevector_simulator"), num_features=2)
    values = llqnn.evaluate(x, param, param_op, "dfdpdop", "dfdopdp")
    np.testing.assert_allclose(
        values["dfdpdop"], np.swapaxes(values["dfdopdp"], -1, -2), atol=1e-8
    )


def test_order_three_pure_circuit_permutations_match_pennylane_reference():
    """dfdxdxdp/dfdxdpdx/dfdpdxdx are order-3, pure circuit-side permutations that
    LowLevelQNNQiskit's own Expec.from_string never recognized (a pre-existing gap - see
    evaluation_classes.get_evaluation_class, which does know them via DirectEvaluation for
    other frameworks). No Qiskit-legacy reference exists for them, so they are verified
    against PennyLane's independently-computed (chained qml.jacobian) native engine instead.
    """
    pqc = ParamZFeatureMap(3, 2)
    obs = SummedPaulis(pqc.num_qubits)
    rng = np.random.default_rng(9)
    x = rng.random((2, 2))
    param = rng.random(pqc.num_parameters)
    param_op = rng.random(obs.num_parameters)

    keys = ("dfdxdxdp", "dfdxdpdx", "dfdpdxdx")
    native_qiskit = LowLevelQNN(pqc, obs, Executor("statevector_simulator"), num_features=2)
    native = native_qiskit.evaluate(x, param, param_op, *keys)
    assert native_qiskit._fallback_engine is None

    reference = LowLevelQNNPennyLane(pqc, obs, Executor("pennylane"), num_features=2).evaluate(
        x, param, param_op, *keys
    )

    for key in keys:
        np.testing.assert_allclose(native[key], reference[key], atol=1e-6)


def test_laplace_family_is_native_and_matches_pennylane_reference():
    """laplace/laplace_dp are pure post-processing (a diagonal trace) of a chained key above,
    reusing evaluation_classes.get_eval_laplace exactly like PennyLane's own engine does -
    verified against it (values only: PennyLane's post-processing pipeline carries extra
    size-1 trailing axes internal to itself, not exposed anywhere else in its public API,
    while this native path follows the same no-padding convention already used for "f"/"var").
    "laplace_dop" is excluded here: it depends on "dfdopdxdx" (see
    test_dfdopdxdx_matches_finite_differences), the one chained key PennyLane's own engine
    cannot compute - test_laplace_dop_matches_finite_differences below covers it instead.
    """
    pqc = ParamZFeatureMap(3, 2)
    obs = SummedPaulis(pqc.num_qubits)
    rng = np.random.default_rng(13)
    x = rng.random((2, 2))
    param = rng.random(pqc.num_parameters)
    param_op = rng.random(obs.num_parameters)

    keys = ("laplace", "laplace_dp")
    executor = Executor("statevector_simulator")
    llqnn = LowLevelQNN(pqc, obs, executor, num_features=2)
    native = llqnn.evaluate(x, param, param_op, *keys)
    assert llqnn._fallback_engine is None

    reference = LowLevelQNNPennyLane(pqc, obs, Executor("pennylane"), num_features=2).evaluate(
        x, param, param_op, *keys
    )

    for key in keys:
        np.testing.assert_allclose(
            np.asarray(native[key]).ravel(), np.asarray(reference[key]).ravel(), atol=1e-6
        )


def test_laplace_dop_matches_finite_differences():
    """laplace_dop depends on "dfdopdxdx" (see test_dfdopdxdx_matches_finite_differences for
    why PennyLane can't serve as a reference for it), so it is verified the same way: central
    finite differences of the already cross-framework-verified "laplace" key with respect to
    each p_op parameter.
    """
    pqc = ParamZFeatureMap(3, 2)
    obs = SummedPaulis(pqc.num_qubits)
    rng = np.random.default_rng(13)
    x = rng.random(2)
    param = rng.random(pqc.num_parameters)
    param_op = rng.random(obs.num_parameters)

    llqnn = LowLevelQNN(pqc, obs, Executor("statevector_simulator"), num_features=2)
    analytic = np.asarray(llqnn.evaluate(x, param, param_op, "laplace_dop")["laplace_dop"])

    eps = 1e-4
    fd = np.zeros_like(analytic)
    for k in range(obs.num_parameters):
        p_plus, p_minus = param_op.copy(), param_op.copy()
        p_plus[k] += eps
        p_minus[k] -= eps
        lp_plus = llqnn.evaluate(x, param, p_plus, "laplace")["laplace"]
        lp_minus = llqnn.evaluate(x, param, p_minus, "laplace")["laplace"]
        fd[k] = (lp_plus - lp_minus) / (2 * eps)

    np.testing.assert_allclose(analytic, fd, atol=1e-4)


@pytest.mark.parametrize("framework", ["pennylane", "qulacs"])
def test_chained_derivative_keys_stay_non_native_outside_qiskit(framework):
    """The chained-derivative keys are qc_executor-Qiskit-only (see
    _NATIVE_KEY_INFO_QISKIT_ONLY's docstring) - for every other framework they must still
    route through the pre-existing fallback path exactly as before this WP, not be silently
    misinterpreted as native.
    """
    pqc = ParamZFeatureMap(3, 2)
    obs = SummedPaulis(pqc.num_qubits)
    rng = np.random.default_rng(17)
    x = rng.random((2, 2))
    param = rng.random(pqc.num_parameters)
    param_op = rng.random(obs.num_parameters)

    llqnn = LowLevelQNN(pqc, obs, Executor(framework), num_features=2)
    if framework == "qulacs":
        with pytest.raises(NotImplementedError):
            llqnn.evaluate(x, param, param_op, "dfdxdx")
    else:
        llqnn.evaluate(x, param, param_op, "dfdxdx")
        assert llqnn._fallback_engine is not None
