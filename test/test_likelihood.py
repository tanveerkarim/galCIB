"""Smoke tests for the rewritten Sampler: it must actually run end-to-end."""
import numpy as np
import pytest

from galCIB.analysis.likelihood import (DEFAULT_PRIORS, FIXED, SAMPLED_PARAMS,
                                        Sampler)


class _FakeCell:
    """Minimal AnalysisModel stand-in: records kwargs, returns fixed shapes."""
    def __init__(self, nell=5, nnu=3):
        self.nell, self.nnu = nell, nnu
        self.last_kwargs = None

    def update_cl(self, **kw):
        self.last_kwargs = kw
        npair = self.nnu * (self.nnu + 1) // 2
        return (np.ones(self.nell),
                np.ones((self.nnu, self.nell)),
                np.ones((npair, self.nell)))


def _make(nell=5, nnu=3, include_cgg=False):
    cell = _FakeCell(nell, nnu)
    n = nnu * nell + (nnu * (nnu + 1) // 2) * nell + (nell if include_cgg else 0)
    data = np.ones(n)
    cov = np.eye(n)
    s = Sampler(cell, data, cov,
                theta_cen=np.array([3.28, 11.49, 0.45, 0.1]),
                theta_sat=np.array([0.38, 10**11.14, 1e13, 0.59]),
                n_nu=nnu, include_cgg=include_cgg)
    return cell, s


def _mid(s):
    lo, hi = s.prior_bounds()
    return 0.5 * (lo + hi)


def test_ndim_and_parser_shapes():
    cell, s = _make()
    assert s.ndim == len(SAMPLED_PARAMS) + 3 + 6
    t_sfr, t_snu, t_IR, sn_gI, sn_II = s.theta_parser(_mid(s))
    assert t_sfr.shape == (5,) and t_snu.shape == (5,) and t_IR.shape == (2,)
    assert sn_gI.shape == (3,) and sn_II.shape == (6,)
    # fixed values must land in the right slots
    assert t_sfr[0] == FIXED["eta_max"]
    assert t_snu[4] == FIXED["gamma_dust"]
    assert t_IR[1] == FIXED["sigma_lnM_IR"]


def test_update_cl_called_with_current_api():
    """The old Sampler passed theta_prof=/bin_cl= and unpacked 4 returns."""
    cell, s = _make()
    s.model_vector(_mid(s))
    kw = cell.last_kwargs
    assert "theta_gal_prof" in kw and "theta_cib_prof" in kw
    assert "theta_prof" not in kw and "bin_cl" not in kw
    assert kw["hmalpha"] == FIXED["hmalpha"]


def test_model_vector_length_matches_data():
    for include_cgg in (False, True):
        cell, s = _make(include_cgg=include_cgg)
        assert s.model_vector(_mid(s)).shape == s.data.shape


def test_prior_rejects_out_of_box_and_bad_size():
    cell, s = _make()
    th = _mid(s)
    assert s.log_prior(th) == 0.0
    bad = th.copy(); bad[0] = DEFAULT_PRIORS["log10Mpeak"][1] + 1
    assert s.log_prior(bad) == -np.inf
    assert s.log_prior(th[:-1]) == -np.inf
    nan = th.copy(); nan[2] = np.nan
    assert s.log_prior(nan) == -np.inf


def test_loglike_and_logpost_finite():
    cell, s = _make()
    th = _mid(s)
    assert np.isfinite(s.loglike(th))
    assert np.isfinite(s.logpost(th))
    # logpost short-circuits outside the prior
    bad = th.copy(); bad[0] = 99.0
    assert s.logpost(bad) == -np.inf


def test_loglike_peaks_when_model_equals_data():
    """model==data must beat a deliberately offset data vector."""
    cell, s = _make()
    th = _mid(s)
    best = s.loglike(th)
    s.data = s.data + 5.0
    assert s.loglike(th) < best


def test_theta_parser_rejects_wrong_dimension():
    cell, s = _make()
    with pytest.raises(ValueError):
        s.theta_parser(np.ones(3))


def test_ell_mask_selects_bandpowers():
    """Regression: the end-to-end run failed because model_vector returned all
    bandpowers while the data used a scale cut. ell_mask must restrict it."""
    nell, nnu = 10, 3
    cell = _FakeCell(nell, nnu)
    mask = np.zeros(nell, bool); mask[2:6] = True      # 4 of 10 bandpowers
    n = nnu * 4 + (nnu * (nnu + 1) // 2) * 4
    s = Sampler(cell, np.ones(n), np.eye(n),
                theta_cen=np.zeros(4), theta_sat=np.ones(4),
                n_nu=nnu, ell_mask=mask)
    assert s.model_vector(_mid(s)).shape == (n,)


def test_mismatched_length_raises_informative_error():
    nell, nnu = 10, 3
    cell = _FakeCell(nell, nnu)
    n = nnu * nell + (nnu * (nnu + 1) // 2) * nell
    s = Sampler(cell, np.ones(n), np.eye(n),
                theta_cen=np.zeros(4), theta_sat=np.ones(4), n_nu=nnu)
    s.ell_mask = np.zeros(nell, bool)      # now selects nothing
    with pytest.raises(ValueError, match="check ell_mask"):
        s.model_vector(_mid(s))
