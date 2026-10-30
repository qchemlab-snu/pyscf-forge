#!/usr/bin/env python
#
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0
#
# Author: Minseok Oh <msjeff2001@snu.ac.kr>
#

'''
GB-srPDFT: grouped-bath CI with a short-range on-top functional, and the
multi-state GB-PDFT variants built on the same model spaces.

State-specific GB-srPDFT on the roots Psi_I of the full-Coulomb GBCI:

    E_I = <Psi_I|H^lr|Psi_I> + E^sr_Hxc[rho_I, Pi_I]

with H^lr the GBCI Hamiltonian with erf-attenuated two-electron integrals
(range separation omega) on the same bath, and E^sr_Hxc the short-range
Hartree plus exchange-correlation energy whose spin polarization comes from
the on-top pair density (`pyscf.gbci.srpdft`).  Pi is the exact one-step
GBCI on-top pair density (``ontop="exact"``) or the active-cumulant
approximation (``ontop="cas_cumulant"``, see `otpd.energy_ot`).

Multi-state variants, all with H^eff_IJ = <Theta_I|H|Theta_J> (full Coulomb)
off the diagonal and the state-specific expression of Theta_I on it:

* XMS: Theta are the XMS intermediate states of the GBCI model space
  (:func:`kernel_xms_gbci`, :func:`kernel_xms_gbpdft`);
* FCD: Theta are fragment-charge-difference (Voityuk-Rosch) diabats, the
  eigenvectors of a fragment occupation operator with rounded eigenvalues,
  H diagonalized inside each charge sector (:func:`kernel_fcd_gbci`,
  :func:`kernel_fcd_xms`).

``*_gbci`` kernels use the GB-srPDFT diagonal, ``*_xms`` / ``*_gbpdft``
kernels the GB-PDFT diagonal of an on-top functional (e.g. tPBE0).

The short-range functional uses the translation of the t (tPBE-type)
on-top functionals (`pyscf.gbci.srpdft`); only that translation is
available.
'''

import numpy as np
from scipy import linalg

from pyscf.dft import gen_grid, numint
from pyscf.fci import cistring
from pyscf.gbci import direct_gbci
from pyscf.gbci import msgbpdft
from pyscf.gbci import otpd as gbci_otpd
from pyscf.gbci import rdm as gbci_rdm
from pyscf.gbci import srpdft
from pyscf.gbci import xmsgbpdft

DEFAULT_OMEGA = 0.4
DEFAULT_SR_X = "GGA_X_PBE_ERF_GWS"
DEFAULT_SR_C = "GGA_C_PBE_ERF_GWS"


def _uncached_range_coulomb(mc, omega):
    """Context manager for erf-attenuated integrals.  The SCF object's
    in-core ``_eri`` is dropped inside the context, since it holds the
    full-Coulomb integrals."""
    mol = mc.mol
    scf_obj = mc._scf

    class _ctx(object):
        def __enter__(self):
            self._range = mol.with_range_coulomb(omega)
            self._range.__enter__()
            self._eri = getattr(scf_obj, "_eri", None)
            scf_obj._eri = None
            return self

        def __exit__(self, *args):
            scf_obj._eri = self._eri
            return self._range.__exit__(*args)

    return _ctx()


def _assert_sr_finite(omega, xc, x_fnal, sr_scale, **arrays):
    """Raise where a non-finite short-range energy is made."""
    for name, a in arrays.items():
        arr = np.asarray(a, dtype=float)
        if not np.isfinite(arr).all():
            raise RuntimeError(
                "short-range %s is not finite at omega=%.3f: %d of %d entries "
                "(xc=%s, x_fnal=%s, sr_scale=%g)"
                % (name, omega, int((~np.isfinite(arr)).sum()), arr.size,
                   xc, x_fnal, sr_scale))


def _clear_gbci_intermediates(mc):
    """Drop the cached GBCI intermediates (their cache key does not see the
    Hamiltonian, e.g. a range-separated one)."""
    if hasattr(mc, "_gbci_intermediates_cache"):
        mc._gbci_intermediates = None
        mc._gbci_intermediates_cache = None


def sr_hxc_energy_ontop(mc, dm1, ci, otpd_data, omega=DEFAULT_OMEGA,
                        xc=DEFAULT_SR_C, x_fnal=DEFAULT_SR_X, grids=None,
                        mo_coeff=None, ot_root_cache=None, report=None,
                        ontop="exact", dm1s=None):
    """Short-range Hartree + xc energy with the spin polarization from Pi.

    t translation (`pyscf.gbci.srpdft`): zeta = 0 wherever R = 4 Pi / rho^2
    >= 1.  ``report``, if given, collects the density with R > 1
    (``electrons_unpolarized``), with Pi < 0 (``electrons_saturated``) and
    evaluated unpolarized after a libxc failure (``electrons_fallback``).

    ``ontop="cas_cumulant"``: Pi = rho_a rho_b + active CAS cumulant (as
    ``otpd.energy_ot``), rho_a, rho_b from ``dm1s`` (dm1 / 2 each when not
    given).
    """
    _otpd = gbci_otpd

    dm1 = np.asarray(dm1)
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ontop not in ("exact", "cas_cumulant"):
        raise ValueError("ontop must be 'exact' or 'cas_cumulant'")
    if ontop == "cas_cumulant":
        from pyscf.mcpdft import otpd as _mcotpd
        if dm1s is None:
            dm1s = (0.5 * dm1, 0.5 * dm1)
        cascm2 = _otpd.make_cas_cumulant(ci, mc.ncas, mc.nelecas)
        mo_cas = mo_coeff[:, mc.ncore:mc.ncore + mc.ncas]
    j_full = np.asarray(mc._scf.get_j(dm=dm1, hermi=1))
    with _uncached_range_coulomb(mc, omega):
        j_lr = np.asarray(mc._scf.get_j(dm=dm1, hermi=1))
    e = 0.5 * float(np.tensordot(j_full - j_lr, dm1))

    if grids is None:
        grids = getattr(mc, "_gbsrpdft_grids", None)
    if grids is None:
        grids = gen_grid.Grids(mc.mol)
        grids.build()
        mc._gbsrpdft_grids = grids
    if ot_root_cache is None and ontop == "exact":
        ot_root_cache = _otpd.make_root_intermediates(
            np.asarray(ci).real, otpd_data)

    ni = numint.NumInt()
    fnals = [f for f in (x_fnal, xc) if f]
    n_sat = n_fb = 0
    rho_sat = rho_fb = rho_scr = 0.0
    for ao, mask, weight, _ in ni.block_loop(mc.mol, grids, mc.mol.nao, 1,
                                             mc.max_memory):
        rho = ni.eval_rho(mc.mol, ao, dm1, mask, "GGA", hermi=1)
        if ontop == "cas_cumulant":
            rhos = np.asarray([ni.eval_rho(mc.mol, ao[0], d, mask, "LDA",
                                           hermi=1) for d in dm1s])
            Pi = np.asarray(_mcotpd.get_ontop_pair_density(
                mc, rhos, ao, cascm2, mo_cas, 0, mask)).ravel()
        else:
            Pi = np.asarray(_otpd.get_ontop_pair_density(
                mc, ao, np.asarray(ci).real, otpd_data, mo_coeff, 0, mask,
                ot_root_cache=ot_root_cache)).ravel()
        u = srpdft.ratio_from_pair_density(rho[0], Pi)
        # a closed-shell core sits at u = 0 up to noise; count only R > 1.01
        # and the Pi < 0 tails
        scr, hi = u < -0.01, u > 1.0
        if scr.any():
            rho_scr += float(np.dot(rho[0][scr], weight[scr]))
        if hi.any():
            n_sat += int(hi.sum())
            rho_sat += float(np.dot(rho[0][hi], weight[hi]))
        for fnal in fnals:
            sub = {}
            exc = srpdft.exc_ontop(fnal, rho, u, omega=omega, report=sub)
            nfb = sub.get("n_unpolarized_fallback", 0)
            if nfb:
                m = sub["fallback_mask"]
                n_fb += nfb
                rho_fb = max(rho_fb, float(np.dot(rho[0][m], weight[m])))
            e += float(np.dot(exc * rho[0], weight))
    _assert_sr_finite(omega, xc, x_fnal, 1.0, sr_hxc_energy_ontop=[e])
    if report is not None:
        report["n_saturated"] = n_sat
        report["electrons_saturated"] = rho_sat
        report["n_fallback"] = n_fb
        report["electrons_fallback"] = rho_fb
        report["electrons_unpolarized"] = rho_scr
    return e


def lr_hamiltonian(mc, mo_coeff, intermediates, omega):
    '''``(h1e, ecore_list, eri)`` of H^lr on a fixed bath.'''
    act = mc.get_active_dm(mo_coeff)
    with _uncached_range_coulomb(mc, omega):
        h1e, ecore = mc.get_h1cas(act, intermediates["mo_list"],
                                  intermediates["dmet_core_list"])
        eri = mc.get_h2eff(mo_coeff)
    return np.asarray(h1e), np.asarray(ecore), np.asarray(eri)


def full_hamiltonian(mc, mo_coeff, intermediates):
    '''``(h1e, ecore_list, eri)`` of the full-Coulomb GBCI Hamiltonian.'''
    act = mc.get_active_dm(mo_coeff)
    h1e, ecore = mc.get_h1cas(act, intermediates["mo_list"],
                              intermediates["dmet_core_list"])
    return (np.asarray(h1e), np.asarray(ecore),
            np.asarray(mc.get_h2eff(mo_coeff)))


def solve(mc, ham, intermediates, nroots, ci0=None):
    h1e, ecore, eri = ham
    if ci0 is not None:
        ci0 = [np.asarray(c).ravel() for c in ci0]
    solver = mc.fcisolver
    saved = solver.nroots
    solver.nroots = nroots
    try:
        e, c = solver.kernel(h1e, eri, mc.ncas, mc.nelecas,
                             intermediates["conf_info_list"],
                             intermediates["ov_list"], ecore, ci0=ci0)
    finally:
        solver.nroots = saved
    ci = list(c) if isinstance(c, (list, tuple)) else [c]
    return np.atleast_1d(np.asarray(e, dtype=float)), ci


def hamiltonian_matrix(mc, ham, intermediates, ci_list):
    '''``<I|H|J>`` for the operator ``ham`` in the basis ``ci_list``.'''
    h1e, ecore, eri = ham
    erieff = mc.fcisolver.absorb_h1e(h1e, eri, mc.ncas, mc.nelecas, 0.5)
    link_index = (cistring.gen_linkstr_index(range(mc.ncas), mc.nelecas[0]),
                  cistring.gen_linkstr_index(range(mc.ncas), mc.nelecas[1]))
    ts = direct_gbci.gen_excitations(mc.ncas, mc.nelecas, link_index=link_index)
    hci = []
    for civec in ci_list:
        hc = msgbpdft._contract_hamiltonian(
            mc.fcisolver, erieff, np.asarray(civec), mc.ncas, mc.nelecas,
            intermediates["conf_info_list"], intermediates["ov_list"],
            ecore, link_index, ts)
        hci.append(np.asarray(hc).ravel())
    n = len(ci_list)
    h = np.empty((n, n))
    for i, bra in enumerate(ci_list):
        for j in range(n):
            h[i, j] = np.vdot(np.asarray(bra).ravel(), hci[j]).real
    return 0.5 * (h + h.T)


class _Context(object):
    '''Everything that depends on the bath only, built once per solve.'''

    def __init__(self, mc, mo_coeff, intermediates, omega, ontop="exact"):
        self.mc = mc
        self.ontop = ontop
        self.mo = mo_coeff
        self.inter = intermediates
        self.omega = float(omega)
        self.rdm_data = gbci_rdm.precompute_rdm1s_data(
            mo_coeff, mc.ncas, mc.nelecas, mc.ncore,
            intermediates["dmet_core_list"], intermediates["conf_info_list"],
            intermediates["ov_list"])
        self.otpd_data = gbci_otpd.make_otpd_intermediates(
            mc.mol.intor("int1e_ovlp"), mo_coeff, mc.ncas, mc.nelecas,
            mc.ncore, intermediates["dmet_core_list"],
            intermediates["conf_info_list"], intermediates["ov_list"])
        self.ham_lr = lr_hamiltonian(mc, mo_coeff, intermediates, omega)
        self._otpd_pdft = None
        self._ham_full = None

    def dm1s(self, civec):
        return np.asarray(gbci_rdm.make_rdm1s_precomputed(
            np.asarray(civec).real, self.rdm_data, mo_coeff=self.mo))

    def dm1(self, civec):
        dm1s = self.dm1s(civec)
        return dm1s[0] + dm1s[1]

    def e_sr(self, civec, report=None):
        c = np.asarray(civec).real
        dm1s = self.dm1s(c)
        return sr_hxc_energy_ontop(
            self.mc, dm1s[0] + dm1s[1], c, self.otpd_data, omega=self.omega,
            mo_coeff=self.mo, report=report, ontop=self.ontop, dm1s=dm1s)

    def e_sr_states(self, ci_list):
        reps = [dict() for _ in ci_list]
        e = np.array([self.e_sr(c, r) for c, r in zip(ci_list, reps)])
        return e, reps

    def ham_full(self):
        if self._ham_full is None:
            self._ham_full = full_hamiltonian(self.mc, self.mo, self.inter)
        return self._ham_full

    def energies(self, ci_list):
        '''<H^lr> + E^sr[rho, Pi] per state.  Returns (energies, <H^lr>,
        E^sr, reports).'''
        e_lr = np.diag(hamiltonian_matrix(self.mc, self.ham_lr, self.inter,
                                          ci_list)).copy()
        e_sr, reps = self.e_sr_states(ci_list)
        return e_lr + e_sr, e_lr, e_sr, reps

    def xms(self, ci_list, ham_ref=None, diag=None):
        '''XMS intermediate states, H^eff, and the adiabatic result.

        Off-diagonal elements from ``ham_ref`` (default H^lr; the kernels
        pass the full-Coulomb Hamiltonian, as XMS-PDFT takes them from the
        reference Hamiltonian).  Diagonal: <Theta_I|H^lr|Theta_I> +
        E^sr[Theta_I], or ``diag(theta)`` when given.
        '''
        if ham_ref is None:
            ham_ref = self.ham_lr
        _, theta = xmsgbpdft.solve_safock(
            self.mc, mo_coeff=self.mo, ci=ci_list, data=self.rdm_data)
        theta = [np.asarray(t) for t in theta]
        heff = hamiltonian_matrix(self.mc, ham_ref, self.inter, theta)
        if diag is None:
            d, _, e_sr, reps = self.energies(theta)
        else:
            d, e_sr, reps = np.asarray([diag(t) for t in theta]), None, None
        heff[np.diag_indices_from(heff)] = d
        e, si = linalg.eigh(heff)
        ci_ad = list(np.tensordot(si.T, np.asarray(theta), axes=1))
        return {"e_states": e, "si": si, "heff": heff, "ci_theta": theta,
                "ci": ci_ad, "e_sr_theta": e_sr, "reports": reps}

    def fcd(self, ci_list, weights, ham_ref, partition="round",
            degen_tol=1e-6, diag=None):
        '''Fragment-charge-difference (Voityuk-Rosch) diabats instead of the
        XMS rotation: eigenvectors of Q_IJ = <Psi_I|N|Psi_J>, N(det) =
        sum_p n_p weights[p]; ``partition="round"`` rounds the eigenvalues
        to integer sectors and diagonalizes ``ham_ref`` inside each sector,
        ``"none"`` keeps every Q eigenvector (``ham_ref`` only inside
        |dq| < ``degen_tol``).  H^eff off-diagonal from ``ham_ref``;
        diagonal <H^lr> + E^sr of each diabat, or ``diag(phi)`` when given.
        '''
        ncas, nelecas = self.mc.ncas, self.mc.nelecas

        w = np.asarray(weights, dtype=float)

        def count(n):
            s = cistring.make_strings(range(ncas), n)
            return np.array([sum(w[p] for p in range(ncas) if (x >> p) & 1)
                             for x in s])
        ndet = count(nelecas[0])[:, None] + count(nelecas[1])[None, :]
        ci = [np.asarray(c) for c in ci_list]
        Q = np.array([[np.vdot(a.ravel(), (ndet * b).ravel()) for b in ci]
                      for a in ci])
        q, U = linalg.eigh(Q)
        if partition == "round":
            sector = np.rint(q).astype(int)
        elif partition == "none":
            sector = np.concatenate([[0], np.cumsum(np.diff(q) >= degen_tol)])
        else:
            raise ValueError("partition must be 'round' or 'none'")
        phi = list(np.tensordot(U.T, np.asarray(ci), axes=1))
        for s in sorted(set(sector.tolist())):
            idx = np.where(sector == s)[0]
            if len(idx) > 1:
                block = [phi[i] for i in idx]
                h = hamiltonian_matrix(self.mc, ham_ref, self.inter, block)
                _, u = linalg.eigh(h)
                sub = np.tensordot(u.T, np.asarray(block), axes=1)
                for k, i in enumerate(idx):
                    phi[i] = sub[k]
        heff = hamiltonian_matrix(self.mc, ham_ref, self.inter, phi)
        if diag is None:
            d, _, e_sr, reps = self.energies(phi)
        else:
            d, e_sr, reps = np.asarray([diag(p) for p in phi]), None, None
        heff[np.diag_indices_from(heff)] = d
        e, si = linalg.eigh(heff)
        n_phi = [float(np.vdot(p.ravel(), (ndet * p).ravel())) for p in phi]
        return {"e_states": e, "si": si, "heff": heff, "ci_theta": phi,
                "ci": list(np.tensordot(si.T, np.asarray(phi), axes=1)),
                "q": q, "sector": sector, "n_diabat": n_phi,
                "e_sr_theta": e_sr, "reports": reps}

    def e_pdft(self, civec, ot=None, ontop="exact"):
        '''GB-PDFT energy of one state; ``ot`` defaults to the GBCI object's.
        ``ontop="cas_cumulant"``: Pi = rho_a rho_b + active CAS cumulant
        (otpd.energy_ot).

        For a hybrid (tPBE0) the wave-function part needs <Psi|H|Psi> of the
        state itself, taken from the full-Coulomb GBCI Hamiltonian.
        '''
        from pyscf.gbci import gbpdft
        if ot is None:
            ot = self.mc.otfnal
        c = np.asarray(civec).real
        dm1s = np.asarray(gbci_rdm.make_rdm1s_precomputed(
            c, self.rdm_data, mo_coeff=self.mo))
        if self._otpd_pdft is None:
            self._otpd_pdft = self.mc.make_otpd_intermediates(
                self.mo, intermediates=self.inter)
        hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, self.mc.mol.spin)[2]
        e_gbci = 0.0
        if abs(hyb[0]) > 1e-10 or abs(hyb[1]) > 1e-10:
            if self._ham_full is None:
                self._ham_full = full_hamiltonian(self.mc, self.mo, self.inter)
            e_gbci = hamiltonian_matrix(self.mc, self._ham_full, self.inter,
                                        [c])[0, 0]
        return (gbpdft.energy_mcwfn(self.mc, self.mo, c, ot=ot, dm1s=dm1s,
                                    e_gbci=e_gbci)
                + gbpdft.energy_dft(self.mc, self.mo, c, ot=ot, dm1s=dm1s,
                                    otpd_data=self._otpd_pdft, ontop=ontop))


def _intermediates(mc, mo_coeff, intermediates):
    if intermediates is None:
        intermediates = mc.get_gbci_intermediates(mo_coeff)
    return intermediates


def kernel_ss(mc, ci_gbci, mo_coeff=None, omega=DEFAULT_OMEGA,
              intermediates=None, ontop="exact"):
    '''State-specific GB-srPDFT on the GBCI roots ``ci_gbci``:
    E_I = <Psi_I|H^lr|Psi_I> + E^sr_Hxc[rho_I, Pi_I].'''
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    inter = _intermediates(mc, mo_coeff, intermediates)
    ctx = _Context(mc, mo_coeff, inter, omega, ontop=ontop)
    e, e_lr, e_sr, reps = ctx.energies([np.asarray(c) for c in ci_gbci])
    return {"e_states": e, "e_lr": e_lr, "e_sr": e_sr, "reports": reps,
            "omega": float(omega)}


def kernel_xms_gbci(mc, ci_gbci, mo_coeff=None, omega=DEFAULT_OMEGA,
                    intermediates=None, ontop="exact"):
    '''XMS-GB-srPDFT: XMS-GBPDFT with E_PDFT replaced by the srPDFT energy.

    The model space is ``ci_gbci``, roots of the full-Coulomb GBCI; the XMS
    rotation is that of XMS-GBPDFT and does not depend on omega.  Then

        H_II = <Theta_I|H^lr|Theta_I> + E^sr_Hxc[rho_I, Pi_I]
        H_IJ = <Theta_I|H|Theta_J>        (full Coulomb, as XMS-PDFT)

    omega -> infinity gives GBCI back.
    '''
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    inter = _intermediates(mc, mo_coeff, intermediates)
    ctx = _Context(mc, mo_coeff, inter, omega, ontop=ontop)
    out = ctx.xms([np.asarray(c) for c in ci_gbci], ham_ref=ctx.ham_full())
    out.update(omega=float(omega))
    return out


def kernel_xms_gbpdft(mc, ci_gbci, mo_coeff=None, intermediates=None,
                      otxc=None, ontop="exact"):
    '''XMS-GBPDFT with the same rotation and off-diagonals as
    :func:`kernel_xms_gbci`; diagonal = E_GB-PDFT[Theta_I].  ``otxc`` (e.g.
    "tPBE0") overrides the GBCI object's on-top functional.'''
    from pyscf import mcpdft
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    ot = None
    if otxc is not None:
        ot = mcpdft.otfnal.get_transfnal(mc.mol, otxc)
        ot.grids = mc.otfnal.grids
    inter = _intermediates(mc, mo_coeff, intermediates)
    ctx = _Context(mc, mo_coeff, inter, DEFAULT_OMEGA)
    ctx._ham_full = full_hamiltonian(mc, mo_coeff, inter)
    return ctx.xms([np.asarray(c) for c in ci_gbci], ham_ref=ctx._ham_full,
                   diag=lambda c: ctx.e_pdft(c, ot=ot, ontop=ontop))


def fragment_weights(mol, mo_coeff, ncore, ncas, atoms):
    '''Lowdin weight of each active MO on the AOs of ``atoms``.'''
    s = mol.intor_symmetric("int1e_ovlp")
    e, v = np.linalg.eigh(s)
    c = ((v * np.sqrt(e)) @ v.T) @ mo_coeff[:, ncore:ncore + ncas]
    aoslice = mol.aoslice_by_atom()
    aos = np.concatenate([np.arange(*aoslice[a, 2:4]) for a in atoms])
    return np.einsum("mp,mp->p", c[aos], c[aos])


def kernel_fcd_xms(mc, ci_gbci, weights, mo_coeff=None, intermediates=None,
                   otxc=None, partition="round", degen_tol=1e-6,
                   ontop="exact"):
    '''XMS-GBPDFT with fragment-charge-difference (Voityuk-Rosch) diabats.

    N(det) = sum_p n_p weights[p] (weights: per active MO, e.g. the donor
    Lowdin weight, or 1 on a virtual MO set for the excitation rank).
    Q_IJ = <Psi_I|N|Psi_J> over the GBCI roots ``ci_gbci``; its eigenvalues
    are rounded to integer sectors and H is diagonalized inside each sector.
    ``partition="none"``: no rounding; every Q eigenvector is its own
    diabat, and H only fixes the rotation inside numerically degenerate Q
    eigenvalues (|dq| < ``degen_tol``).
    H^eff diagonal: GB-PDFT of each diabat (``otxc`` overrides the
    functional); off-diagonal: full-Coulomb GBCI.  Depends on the model
    space, i.e. on nroots.
    '''
    from pyscf import mcpdft
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    ot = None
    if otxc is not None:
        ot = mcpdft.otfnal.get_transfnal(mc.mol, otxc)
        ot.grids = mc.otfnal.grids
    inter = _intermediates(mc, mo_coeff, intermediates)
    ctx = _Context(mc, mo_coeff, inter, DEFAULT_OMEGA)
    return ctx.fcd(ci_gbci, weights, ctx.ham_full(), partition=partition,
                   degen_tol=degen_tol,
                   diag=lambda p: ctx.e_pdft(p, ot=ot, ontop=ontop))


def kernel_fcd_gbci(mc, ci_gbci, weights, mo_coeff=None,
                    omega=DEFAULT_OMEGA, intermediates=None,
                    partition="round", degen_tol=1e-6, ontop="exact"):
    '''FCD-GB-srPDFT: :func:`kernel_xms_gbci` with the XMS rotation replaced
    by fragment-charge-difference diabats (:func:`kernel_fcd_xms`).

        H_II = <Phi_I|H^lr|Phi_I> + E^sr_Hxc[rho_I, Pi_I]
        H_IJ = <Phi_I|H|Phi_J>        (full Coulomb)
    '''
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    inter = _intermediates(mc, mo_coeff, intermediates)
    ctx = _Context(mc, mo_coeff, inter, omega, ontop=ontop)
    out = ctx.fcd(ci_gbci, weights, ctx.ham_full(), partition=partition,
                  degen_tol=degen_tol)
    out.update(omega=float(omega))
    return out
