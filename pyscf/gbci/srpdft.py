#!/usr/bin/env python
#
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0
#
# Author: Minseok Oh <msjeff2001@snu.ac.kr>
#
'''Short-range exchange-correlation functional on the on-top pair density.

The short-range functional (erf-attenuated PBE) takes its spin polarization
from the on-top pair density, as in MC-srPDFT (Jensen et al., JCP 162,
034104 (2025)), by the translation of the t (tPBE-type) on-top functionals:

    zeta(r) = sqrt(1 - R),  R = 4 Pi(r) / rho(r)^2   for R < 1,
    zeta(r) = 0                                       for R >= 1.

Only this translation is available (no fully translated, ft, form).
'''

import numpy as np

from pyscf.dft import libxc

RHO_CUTOFF = 1e-12
# libxc's erf functionals return NaN at isolated float-level points (one grid
# point in ~1e6, rho ~ 5e-10); those points are recomputed at rho * (1 + NUDGE)
NUDGE = 1e-12


def eval_xc_finite(fnal, rho, spin=0, deriv=1, omega=None, on_fail="raise"):
    """``libxc.eval_xc`` with its isolated NaN points repaired.

    Points whose output is non-finite are recomputed at ``rho * (1 + NUDGE)``
    (every component).  Points that stay non-finite raise
    FloatingPointError, or are returned as NaN with ``on_fail="nan"``.
    """
    rho = np.asarray(rho, dtype=float)
    out = libxc.eval_xc(fnal, rho, spin=spin, deriv=deriv, omega=omega)
    exc = out[0]
    arrays = [exc] + [a for a in (out[1] if deriv > 0 else ()) if a is not None]
    bad = np.zeros(exc.shape[0], dtype=bool)
    for a in arrays:
        a = np.asarray(a)
        fin = np.isfinite(a) if a.ndim == 1 else np.isfinite(a).all(axis=1)
        bad |= ~fin
    if bad.any():
        sub = libxc.eval_xc(fnal, rho[..., bad] * (1.0 + NUDGE), spin=spin,
                            deriv=deriv, omega=omega)
        exc[bad] = sub[0]
        if deriv > 0:
            for a, b in zip(out[1], sub[1]):
                if a is not None:
                    a[bad] = b
        still = np.zeros(exc.shape[0], dtype=bool)
        for a in [exc] + [a for a in (out[1] if deriv > 0 else ()) if a is not None]:
            a = np.asarray(a)
            fin = np.isfinite(a) if a.ndim == 1 else np.isfinite(a).all(axis=1)
            still |= ~fin
        if still.any() and on_fail == "raise":
            i = int(np.argmax(still))
            raise FloatingPointError(
                "%s (spin=%d, omega=%s) is non-finite at %d point(s) even after "
                "a %g relative nudge; first: rho[..., %d] = %r"
                % (fnal, spin, omega, int(still.sum()), NUDGE, i,
                   rho[..., i].tolist()))
    return out


def _rho_spin(rho, zeta):
    '''Spin-resolved GGA density from a spin-summed one and zeta (tPBE
    translation: the gradient of zeta is dropped).'''
    out = np.zeros((2,) + rho.shape)
    half = 0.5 * rho
    out[0] = half * (1.0 + zeta)
    out[1] = half * (1.0 - zeta)
    return out


def ratio_from_pair_density(rho, Pi, tol=1e-12):
    '''``u = 1 - 4 Pi / rho^2``, guarded where the density vanishes.

    Where rho is below ``tol`` u is set to 0 (unpolarized).
    '''
    rho = np.asarray(rho, dtype=float)
    Pi = np.asarray(Pi, dtype=float)
    u = np.zeros_like(rho)
    ok = rho > tol
    u[ok] = 1.0 - 4.0 * Pi[ok] / rho[ok] ** 2
    return u


def exc_ontop(fnal, rho, u, omega=None):
    '''Exchange-correlation energy per particle at the translated on-top
    polarization.

    ``rho`` is ``(4, N)`` in PySCF's GGA layout and spin summed; ``u`` is
    ``1 - 4 Pi / rho^2`` per point.  u is clipped to [0, 1] and libxc is
    called at zeta = sqrt(u).  Points where the spin-polarized libxc
    call stays non-finite fall back to the unpolarized value.
    '''
    rho = np.asarray(rho, dtype=float)
    if rho.ndim == 1:
        rho = rho[None, :]
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    out = np.zeros(u.size)
    live = rho[0] > RHO_CUTOFF
    if live.any():
        out[live] = eval_xc_finite(
            fnal, _rho_spin(rho[:, live], np.sqrt(u[live])), spin=1, deriv=0,
            omega=omega, on_fail="nan")[0]
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = eval_xc_finite(fnal, rho[:, bad], spin=0, deriv=0,
                                  omega=omega)[0]
    return out
