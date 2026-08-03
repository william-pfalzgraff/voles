"""Tests for the deterministic Gauss-Jacobi singular-block quadrature.

Covers: the folded-weight rule itself (exactness against closed-form Beta
integrals), the {location: alpha} / (location, alpha) API forms, the
acceptance-check fallback (wrong exponent degrades bitwise to the adaptive
path), weight-tensor accuracy against tight adaptive references (scalar and
vector), and end-to-end solves for all three function solvers.
"""
import numpy as np
import pytest

from voles import (function_solve_VIE_1, function_solve_VIE_2,
                   function_solve_VIDE, optimal_graded_mesh)
from voles._callable_solvers import (_gauss_jacobi_nodes_weights,
                                     _normalize_kernel_singularity,
                                     _build_W_scalar, _build_W_vector)

from conftest import (TOLERANCE, as_callable,
                      VIE1_SPEC_ABEL, VIE2_SPEC_ABEL, VIDE_SPEC_ABEL)


# ---------------------------------------------------------------------------
# The rule itself: half * sum(w_folded * F(s_q)) must reproduce closed-form
# Beta integrals of (B-s)^{-aR} (s-A)^{-aL} P(s) exactly for polynomial P.
# ---------------------------------------------------------------------------

def _gj_apply(order, A, B, aR, aL, F):
    x, wf = _gauss_jacobi_nodes_weights(order, aR, aL)
    half = 0.5 * (B - A)
    s_q = 0.5 * (A + B) + half * x
    return half * np.sum(wf * F(s_q))


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.9])
@pytest.mark.parametrize("k", [0, 3, 7, 11])
def test_rule_exact_right_singularity(alpha, k):
    """int_A^B (B-s)^{-alpha} (s-A)^k ds = (B-A)^{k+1-alpha} B(k+1, 1-alpha),
    exact for the order-6 rule up to polynomial degree 11."""
    from scipy.special import beta
    A, B = 0.3, 0.9
    exact = (B - A) ** (k + 1 - alpha) * beta(k + 1, 1 - alpha)
    got = _gj_apply(6, A, B, alpha, 0.0,
                    lambda s: (B - s) ** (-alpha) * (s - A) ** k)
    assert abs(got - exact) <= 1e-13 * abs(exact)


@pytest.mark.parametrize("aR,aL", [(0.5, 0.5), (0.3, 0.6)])
@pytest.mark.parametrize("k", [0, 2, 5])
def test_rule_exact_both_singularities(aR, aL, k):
    """Two-sided weight: int (B-s)^{-aR} (s-A)^{-aL} (s-A)^k ds
    = (B-A)^{1-aR-aL+k} B(1-aL+k, 1-aR)."""
    from scipy.special import beta
    A, B = 0.0, 0.7
    exact = (B - A) ** (1 - aR - aL + k) * beta(1 - aL + k, 1 - aR)
    got = _gj_apply(6, A, B, aR, aL,
                    lambda s: (B - s) ** (-aR) * (s - A) ** (-aL) * (s - A) ** k)
    assert abs(got - exact) <= 1e-13 * abs(exact)


def test_rule_zero_exponents_is_gauss_legendre():
    """(0, 0) folding reduces to plain Gauss-Legendre."""
    x_gj, w_gj = _gauss_jacobi_nodes_weights(6, 0.0, 0.0)
    x_gl, w_gl = np.polynomial.legendre.leggauss(6)
    assert np.allclose(x_gj, x_gl, atol=1e-14)
    assert np.allclose(w_gj, w_gl, atol=1e-14)


# ---------------------------------------------------------------------------
# API normalization
# ---------------------------------------------------------------------------

def test_normalize_dict_and_pair_forms():
    for form in ({0.0: 0.5}, [(0.0, 0.5)]):
        entries = _normalize_kernel_singularity(form)(2.0)
        assert entries == [(2.0, 0.5)]


def test_normalize_mixed_list():
    entries = _normalize_kernel_singularity([0.0, (0.5, 0.3)])(2.0)
    assert entries == [(2.0, None), (1.5, 0.3)]


def test_normalize_bare_tuple_is_two_locations():
    """Backward compat: a flat tuple of floats is two locations, not a
    (location, alpha) pair -- pairs must be nested, e.g. [(0.0, 0.5)]."""
    entries = _normalize_kernel_singularity((0.0, 0.5))(2.0)
    assert entries == [(2.0, None), (1.5, None)]


def test_normalize_callable_locations_only():
    entries = _normalize_kernel_singularity(lambda t: [t, t - 0.5])(2.0)
    assert entries == [(2.0, None), (1.5, None)]


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 2.0])
def test_invalid_alpha_raises(bad):
    with pytest.raises(ValueError, match="0 < alpha < 1"):
        _normalize_kernel_singularity({0.0: bad})


def test_invalid_singular_quadrature_raises(vie2_callable_abel):
    p = vie2_callable_abel
    with pytest.raises(ValueError, match="singular_quadrature"):
        function_solve_VIE_2(kernel=p["kernel"], g=p["g"],
                             mesh_breakpoints=np.linspace(0, 1, 6),
                             coll_divs=p["coll_divs"],
                             coll_choices=p["coll_choices"],
                             kernel_singularity={0.0: 0.5},
                             singular_quadrature="bogus")


# ---------------------------------------------------------------------------
# Policy: escape hatch and fallback reproduce the adaptive path bitwise
# ---------------------------------------------------------------------------

def _abel_common(spec, mesh):
    p = as_callable(spec, coll_divs=2, coll_choices=[0, 1, 2])
    return dict(kernel=p["kernel"], g=p["g"], mesh_breakpoints=mesh,
                coll_divs=2, coll_choices=[0, 1, 2], show_warnings=False)


def test_adaptive_escape_hatch_matches_bare_bitwise():
    common = _abel_common(VIE2_SPEC_ABEL, np.linspace(0, 1, 21))
    y_bare = function_solve_VIE_2(kernel_singularity=0.0, **common)
    y_off = function_solve_VIE_2(kernel_singularity={0.0: 0.5},
                                 singular_quadrature='adaptive', **common)
    assert np.array_equal(y_bare, y_off)


def test_wrong_alpha_falls_back_bitwise():
    """alpha=0.25 declared on a u^{-1/2} kernel: the regularized integrand is
    still singular, the two-order check fails on every singular block, and
    the fallback reproduces the bare-location adaptive path bitwise."""
    common = _abel_common(VIE2_SPEC_ABEL, np.linspace(0, 1, 21))
    y_bare = function_solve_VIE_2(kernel_singularity=0.0, **common)
    y_wrong = function_solve_VIE_2(kernel_singularity={0.0: 0.25}, **common)
    assert np.array_equal(y_bare, y_wrong)


def test_smooth_kernel_unaffected_by_mode(vie2_callable_smooth):
    """No declared singularity: singular_quadrature must be a no-op."""
    p = vie2_callable_smooth
    common = dict(kernel=p["kernel"], g=p["g"],
                  mesh_breakpoints=np.linspace(0, 1, 21),
                  coll_divs=p["coll_divs"], coll_choices=p["coll_choices"])
    y_auto = function_solve_VIE_2(singular_quadrature='auto', **common)
    y_adap = function_solve_VIE_2(singular_quadrature='adaptive', **common)
    assert np.array_equal(y_auto, y_adap)


# ---------------------------------------------------------------------------
# Weight-tensor accuracy against tight adaptive references
# ---------------------------------------------------------------------------

def test_scalar_weights_match_tight_adaptive():
    """K(u) = u^{-1/2} (2 + cos u): smooth factor times the declared power.
    GJ weights must agree with epsabs=epsrel=1e-12 adaptive quadrature far
    below the 1e-9 acceptance tolerance."""
    K = lambda u: (2.0 + np.cos(u)) / np.sqrt(u) if u > 0 else 0.0
    mesh = np.linspace(0.0, 1.0, 9)
    node_pos = np.array([0.0, 0.5, 1.0])
    W_gj = _build_W_scalar(K, mesh, node_pos, {0.0: 0.5}, 6)
    W_ref = _build_W_scalar(K, mesh, node_pos, 0.0, 6,
                            reuse_adaptive_blocks=True)  # tight-tol adaptive
    scale = np.max(np.abs(W_ref))
    assert np.max(np.abs(W_gj - W_ref)) <= 1e-10 * scale


def test_interior_singularity_gj_engages():
    """Pure two-sided kernel |u - 0.5|^{-1/2} (2 + cos u): interior singular
    blocks split at s* = tau - 0.5 and both one-sided GJ pieces must pass the
    check (agreement with the tight adaptive reference well below 1e-9)."""
    K = lambda u: (2.0 + np.cos(u)) / np.sqrt(abs(u - 0.5)) \
        if abs(u - 0.5) > 1e-14 else 0.0
    mesh = np.linspace(0.0, 1.0, 9)
    node_pos = np.array([0.0, 0.5, 1.0])
    W_gj = _build_W_scalar(K, mesh, node_pos, {0.5: 0.5}, 6)
    W_ref = _build_W_scalar(K, mesh, node_pos, 0.5, 6,
                            reuse_adaptive_blocks=True)
    scale = np.max(np.abs(W_ref))
    assert np.max(np.abs(W_gj - W_ref)) <= 1e-10 * scale


def test_vector_weights_match_scalar_blocks():
    """Vector GJ path: K(u) = k(u) * C for a constant (2, 2) matrix C must
    reproduce the scalar GJ weights entrywise (W_vec[..., i, j] = C_ij * W_sc).
    This checks the vector path without inheriting quad_vec's ~1e-7 errors."""
    C = np.array([[2.0, 1.0], [0.5, 3.0]])
    k = lambda u: 1.0 / np.sqrt(u) if u > 0 else 0.0
    K_vec = lambda u: k(u) * C
    mesh = np.linspace(0.0, 1.0, 9)
    node_pos = np.array([0.0, 0.5, 1.0])
    W_sc = _build_W_scalar(k, mesh, node_pos, {0.0: 0.5}, 6)
    W_vec = _build_W_vector(K_vec, mesh, node_pos, {0.0: 0.5}, 6, 2)
    for i in range(2):
        for j in range(2):
            assert np.allclose(W_vec[..., i, j], C[i, j] * W_sc,
                               atol=1e-13 * np.max(np.abs(W_sc)))


# ---------------------------------------------------------------------------
# End-to-end solves
# ---------------------------------------------------------------------------

def test_vie2_abel_gj_solves_exactly():
    """Graded mesh (the suite's Abel pattern) against the exact solution --
    the sqrt(t) layer needs the graded mesh regardless of how the weights
    are computed -- plus a uniform-mesh consistency check: GJ weights must
    reproduce the adaptive solve to the acceptance-check level."""
    mesh = np.linspace(0, 1, 41) ** 3
    common = _abel_common(VIE2_SPEC_ABEL, mesh)
    y = function_solve_VIE_2(kernel_singularity={0.0: 0.5}, **common)
    taus = mesh[:-1, None] + np.array([0, 0.5, 1.0]) * np.diff(mesh)[:, None]
    assert np.max(np.abs(y - np.sqrt(taus))) < TOLERANCE

    common_u = _abel_common(VIE2_SPEC_ABEL, np.linspace(0, 1, 41))
    y_gj = function_solve_VIE_2(kernel_singularity={0.0: 0.5}, **common_u)
    y_ad = function_solve_VIE_2(kernel_singularity=0.0, **common_u)
    assert np.max(np.abs(y_gj - y_ad)) <= 1e-8 * np.max(np.abs(y_ad))


def test_vie1_abel_gj_solves_exactly():
    p = as_callable(VIE1_SPEC_ABEL, coll_divs=3, coll_choices=[1, 2, 3])
    mesh = np.linspace(0, 1, 41)
    y = function_solve_VIE_1(kernel=p["kernel"], g=p["g"],
                             mesh_breakpoints=mesh, coll_divs=3,
                             coll_choices=[1, 2, 3],
                             kernel_singularity={0.0: 0.5},
                             show_warnings=False)
    taus = mesh[:-1, None] + (np.array([1, 2, 3]) / 3.0) * np.diff(mesh)[:, None]
    assert np.max(np.abs(y - np.sqrt(taus))) < TOLERANCE * 10


def test_vide_abel_gj_solves_exactly():
    p = as_callable(VIDE_SPEC_ABEL, coll_divs=2, coll_choices=[0, 1, 2])
    mesh = np.linspace(0, 1, 41)
    y = function_solve_VIDE(kernel=p["kernel"], a=p["a"], g=p["g"],
                            soln_init_value=p["soln_init_value"],
                            mesh_breakpoints=mesh, coll_divs=2,
                            coll_choices=[0, 1, 2],
                            kernel_singularity={0.0: 0.5},
                            show_warnings=False)
    taus = mesh[:-1, None] + np.array([0, 0.5, 1.0]) * np.diff(mesh)[:, None]
    assert np.max(np.abs(y - taus ** 1.5)) < TOLERANCE * 10


def test_gj_on_graded_mesh_matches_adaptive():
    """Off the Toeplitz fast path (graded mesh) GJ still applies per row;
    values agree with the adaptive path to well below the check tolerance."""
    mesh = optimal_graded_mesh(alpha=0.5, T=1.0, M=20, order=3)
    common = _abel_common(VIE2_SPEC_ABEL, mesh)
    y_bare = function_solve_VIE_2(kernel_singularity=0.0, **common)
    y_gj = function_solve_VIE_2(kernel_singularity={0.0: 0.5}, **common)
    scale = np.max(np.abs(y_bare))
    assert np.max(np.abs(y_gj - y_bare)) <= 1e-8 * scale


def test_gj_with_reuse_flag_consistent():
    """reuse_adaptive_blocks + declared exponents: GJ handles the singular
    blocks (nothing left to reuse) and the result matches the plain GJ solve
    to rounding."""
    common = _abel_common(VIE2_SPEC_ABEL, np.linspace(0, 1, 21))
    y_gj = function_solve_VIE_2(kernel_singularity={0.0: 0.5}, **common)
    y_both = function_solve_VIE_2(kernel_singularity={0.0: 0.5},
                                  reuse_adaptive_blocks=True, **common)
    scale = np.max(np.abs(y_gj))
    assert np.max(np.abs(y_both - y_gj)) <= 1e-12 * scale


def test_gj_complex_kernel():
    """Complex Abel kernel through the block-decomposition dispatch."""
    K = lambda u: (1.0 + 0.5j) / np.sqrt(u) if u > 0 else 0.0
    g = lambda t: np.sqrt(t) - (1.0 + 0.5j) * 0.5 * np.pi * t
    mesh = np.linspace(0, 1, 21)
    common = dict(kernel=K, g=g, mesh_breakpoints=mesh,
                  coll_divs=2, coll_choices=[0, 1, 2], show_warnings=False)
    y_bare = function_solve_VIE_2(kernel_singularity=0.0, **common)
    y_gj = function_solve_VIE_2(kernel_singularity={0.0: 0.5}, **common)
    assert np.all(np.isfinite(y_gj.real)) and np.all(np.isfinite(y_gj.imag))
    scale = np.max(np.abs(y_bare))
    assert np.max(np.abs(y_gj - y_bare)) <= 1e-5 * scale
