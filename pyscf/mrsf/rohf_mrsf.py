#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Minseok Oh <msjeff2001@snu.ac.kr>
#         Nakhyun Kim <kimnh97@snu.ac.kr>
#
'''
Mixed-reference spin-flip (MRSF) and extended MRSF (EMRSF) TDA for a triplet ROHF/ROKS
reference, matrix-free in the AO basis. EMRSF adds the core -> virtual configurations of
the closed-shell singlet reference (O1 doubly occupied) to the MRSF response space.

The SCF object supplies the integrals (hcore, J/K, XC), so molecules and Gamma-point PBC
cells are both supported. The response space is spanned by mo_coeff (default: all
orbitals of mf); doubly occupied orbitals in mo_core are frozen and enter only through
the reference density (embedding). Integrals already in the MO basis of the response
orbitals can also be given directly (TDA_MRSF.given_mo_ints).

Refs:
    S. Lee, M. Filatov, S. Lee and C. H. Choi, J. Chem. Phys. 149, 104101 (2018)
    M. Oh, N. Kim, Y. Jung, C. H. Choi, S. Lee, J. Chem. Theory Comput. 22, 6057 (2026)
'''

import numpy
from pyscf import lib, gto, scf, ao2mo
from pyscf.lib import logger
from pyscf.data.nist import HARTREE2EV
from pyscf.tdscf.rhf import TDBase
from pyscf.dft.rks import KohnShamDFT
from pyscf import __config__

SQRT2 = numpy.sqrt(2.)
MO_BASE = getattr(__config__, 'MO_BASE', 1)


def order_orbitals(mo_coeff, mo_occ):
    '''Orbitals reordered as doubly occupied, singly occupied, empty, keeping the order
    within each group (e.g. after MOM). Returns (mo_coeff, mo_occ, perm), perm None if the
    order is unchanged.'''
    occ = numpy.asarray(mo_occ, dtype=float)
    if occ.ndim != 1:
        raise ValueError('MRSF needs a restricted open-shell (ROHF/ROKS) reference')
    perm = numpy.concatenate([numpy.where(occ == k)[0] for k in (2, 1, 0)])
    if perm.size != occ.size:
        raise ValueError('mo_occ must contain only 2, 1 and 0')
    if numpy.all(perm == numpy.arange(occ.size)):
        return numpy.asarray(mo_coeff), occ, None
    return numpy.asarray(mo_coeff)[:, perm], occ[perm], perm


def hybrid_coeff(mf):
    '''Fraction of exact exchange in mf.xc (1 for HF).'''
    if isinstance(mf, KohnShamDFT):
        return mf._numint.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)[2]
    return 1.


def _check_hyb(mf, hyb, extended, fock):
    '''Range-separated functionals are not supported. A hyb other than the exact-exchange
    fraction of mf (given_mo_ints) is consistent only for MRSF with given Fock matrices:
    the response kernel is hyb*K, while the EMRSF CV block takes its kernel from mf.'''
    if isinstance(mf, KohnShamDFT):
        if mf._numint.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)[0] != 0:
            raise NotImplementedError('MRSF/EMRSF does not support range-separated functionals')
    if abs(hyb - hybrid_coeff(mf)) > 1e-12 and (extended or fock is None):
        raise NotImplementedError(
            'a hyb other than that of mf needs fock=(focka, fockb) and extended=False '
            '(the EMRSF CV block takes its kernel from mf)')


def mf_given_mo_ints(h1e, eri, mo_occ, e_core=0.):
    '''ROHF object for integrals given in an orthonormal MO basis, which serves as its AO
    basis (overlap = 1, mo_coeff = 1): one-electron Hamiltonian h1e (frozen-core potential
    included), ERIs eri ((n,n,n,n) or any form ao2mo.restore accepts) and constant energy
    e_core. mo_occ is set, so it can be passed to TDA_MRSF directly.'''
    h1e = numpy.asarray(h1e)
    n = h1e.shape[0]
    occ = numpy.asarray(mo_occ, dtype=float)
    mol = gto.M(verbose=0)
    mol.nelectron = int(round(occ.sum()))
    mol.spin = int(numpy.count_nonzero(occ == 1))
    mol.incore_anyway = True
    mol.nao_nr = lambda *args: n
    mf = scf.ROHF(mol)
    mf._eri = ao2mo.restore(8, numpy.asarray(eri), n)
    mf.get_hcore = lambda *args: h1e
    mf.get_ovlp = lambda *args: numpy.eye(n)
    mf.energy_nuc = lambda *args: e_core
    mf.mo_coeff = numpy.eye(n)
    mf.mo_occ = occ
    mf.mo_energy = numpy.zeros(n)
    mf.converged = True
    return mf


def mf_embedded(mf, mo_coeff, mo_occ, mo_core=None, fock=None):
    '''Triplet reference of the response space (frozen core included).

    Returns (mfe, focka, fockb, e_ref):
        mfe     copy of mf whose mo_coeff/mo_occ are the response orbitals (it supplies the
                AO integrals for the response),
        focka, fockb  alpha/beta Fock matrices in the mo_coeff basis (fock if given),
        e_ref   total energy of the core + response-space triplet determinant.
    The Fock matrices and e_ref are built from the full density (frozen core included).'''
    mo_coeff = numpy.asarray(mo_coeff)
    mo_occ = numpy.asarray(mo_occ, dtype=float)
    nao = mo_coeff.shape[0]
    mo_core = numpy.zeros((nao, 0)) if mo_core is None else numpy.asarray(mo_core)
    if mo_core.ndim != 2 or mo_core.shape[0] != nao:
        raise ValueError('mo_core must be an (nao, ncore) array')

    mfe = mf.copy()
    mfe.mo_coeff = mo_coeff
    mfe.mo_occ = mo_occ
    mfe.mo_energy = numpy.zeros(mo_occ.size)
    ca = mo_coeff[:, mo_occ > 0]
    cb = mo_coeff[:, mo_occ > 1]
    dmc = mo_core.dot(mo_core.T)
    dm = numpy.asarray((dmc + ca.dot(ca.T), dmc + cb.dot(cb.T)))
    # UHF/UKS form of the same integrals gives the spin-resolved Fock (and XC) directly
    mfu = mfe.to_uks() if isinstance(mf, KohnShamDFT) else mfe.to_uhf()
    h1e = mf.get_hcore()
    vhf = mfu.get_veff(mf.mol, dm)
    fock_ao = numpy.asarray((h1e + vhf[0], h1e + vhf[1]))
    e_ref = mfu.energy_tot(dm, h1e, vhf)
    if fock is None:
        focka = mo_coeff.T.dot(fock_ao[0]).dot(mo_coeff)
        fockb = mo_coeff.T.dot(fock_ao[1]).dot(mo_coeff)
    else:
        focka, fockb = numpy.asarray(fock[0]), numpy.asarray(fock[1])
    mfe.mo_energy = .5 * (focka.diagonal() + fockb.diagonal())
    return mfe, focka, fockb, e_ref


def orb_indices(mo_occ):
    '''Doubly occupied, open (O1, O2) and empty orbital indices for occupations ordered
    as 2, ..., 2, 1, 1, 0, ..., 0.'''
    nc = numpy.count_nonzero(numpy.asarray(mo_occ) == 2)
    return numpy.arange(nc), numpy.arange(nc, nc + 2), numpy.arange(nc + 2, len(mo_occ))


def pairs(a, b):
    return numpy.array([(p, q) for p in a for q in b], dtype=int).reshape(-1, 2)


def _open_jk(mf, oidx):
    '''MO J_ab = (ab|xy) and K_ab = (ax|by) for the open-open densities C_a C_b^T.'''
    c = mf.mo_coeff
    co = c[:, oidx]
    keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
    dms = numpy.array([numpy.outer(co[:, a], co[:, b]) for a, b in keys])
    vj, vk = mf.get_jk(mf.mol, dms, hermi=0)
    jmo = dict((k, c.T.dot(vj[i]).dot(c)) for i, k in enumerate(keys))
    kmo = dict((k, c.T.dot(vk[i]).dot(c)) for i, k in enumerate(keys))
    return jmo, kmo


def _occ_diag_jk(mf, nocc):
    '''(pp|xx) and (px|px) for the occupied response orbitals p (core + open) and all
    response orbitals x, from one J/K evaluation with the densities C_p C_p^T. Exact and
    independent of how mf evaluates J/K (molecule or PBC, direct, incore or fitted).
    Used only for hdiag (initial guess and preconditioner).'''
    c = mf.mo_coeff
    co = c[:, :nocc]
    dms = numpy.einsum('mp,np->pmn', co, co)
    vj, vk = mf.get_jk(mf.mol, dms, hermi=1)
    jdiag = lib.einsum('mx,pmn,nx->px', c, numpy.asarray(vj), c)
    kdiag = lib.einsum('mx,pmn,nx->px', c, numpy.asarray(vk), c)
    return jdiag, kdiag


def spin_pair_block(kmo, hyb, cidx, oidx, vidx):
    '''Spin-pairing coupling (Slater-Condon) in the CV + OV + CO basis from open-open K.'''
    nc, nv = len(cidx), len(vidx)
    ncv = nc * nv
    n = ncv + 2 * nv + 2 * nc
    ov = [slice(ncv + k * nv, ncv + (k + 1) * nv) for k in range(2)]
    co = [slice(ncv + 2 * nv + k, n, 2) for k in range(2)]
    vv, cc = numpy.ix_(vidx, vidx), numpy.ix_(cidx, cidx)
    cmat = numpy.zeros((n, n))
    for k in range(2):
        kp = 1 - k
        cmat[ov[k], ov[k]] = -kmo[kp, kp][vv].T
        cmat[co[k], co[k]] = -kmo[kp, kp][cc].T
        cmat[ov[k], ov[kp]] = kmo[kp, k][vv].T
        cmat[co[k], co[kp]] = kmo[k, kp][cc]
    x = (kmo[0, 1] - kmo[1, 0])[numpy.ix_(vidx, cidx)]
    cmat[ov[0], co[1]] = -x
    cmat[ov[1], co[0]] = x
    cmat[co[1], ov[0]] = -x.T
    cmat[co[0], ov[1]] = x.T
    return hyb * cmat


def _sf_vind(mf, focka, fockb, hyb):
    '''Collinear spin-flip TDA operator (alpha occupied -> beta empty) of the response
    orbitals mf.mo_coeff: Fock part plus -hyb K of the AO transition densities (the
    collinear kernel has no XC part). focka, fockb: Fock matrices in the mo_coeff basis.'''
    c = mf.mo_coeff
    occ = mf.mo_occ
    o, v = occ > 0, occ < 2
    orbo, orbv = c[:, o], c[:, v]
    focko = numpy.asarray(focka)[numpy.ix_(o, o)]
    fockv = numpy.asarray(fockb)[numpy.ix_(v, v)]
    nao, nocc, nvir = c.shape[0], orbo.shape[1], orbv.shape[1]

    def vind(zs):
        zs = numpy.asarray(zs).reshape(-1, nocc, nvir)
        v1mo = lib.einsum('ab,xib->xia', fockv, zs) - lib.einsum('ji,xja->xia', focko, zs)
        if hyb != 0:
            dms = lib.einsum('xov,pv,qo->xpq', zs, orbv, orbo)
            vk = numpy.asarray(mf.get_k(mf.mol, dms, hermi=0)).reshape(-1, nao, nao)
            v1mo -= hyb * lib.einsum('xpq,qo,pv->xov', vk, orbo, orbv)
        return v1mo.reshape(len(zs), -1)
    return vind


def gen_mrsf_vind(mf, hyb, spc, focka, fockb, jmo, kmo, cidx, oidx, vidx, occ_diag=None,
                  singlet=True):
    '''MRSF part: the collinear SF-TDA operator projected onto [G, D, L-R] + CV + OV + CO
    (singlet) or [L+R] + CV + OV + CO (triplet), plus the spin-pairing coupling.'''
    o1, o2 = oidx
    nc, nv = len(cidx), len(vidx)
    nvirb = nv + 2
    vind_sf = _sf_vind(mf, focka, fockb, hyb)
    xi = lambda p, q: p * nvirb + (q - nc)
    cols = numpy.vstack((pairs(cidx, vidx), pairs(oidx, vidx), pairs(cidx, oidx)))
    i456 = xi(cols[:, 0], cols[:, 1])
    i_g, i_d, i_11, i_22 = xi(o2, o1), xi(o1, o2), xi(o1, o1), xi(o2, o2)
    noo = 3 if singlet else 1
    nm = noo + len(cols)
    # the spin-pairing coupling enters with - for singlets and + for triplets
    cmat = spin_pair_block(kmo, hyb, cidx, oidx, vidx) * (-1 if singlet else 1) if spc else None

    hd = numpy.empty(nm)
    lr_diag = (.5 * (fockb[o1, o1] - focka[o1, o1] - hyb * jmo[0, 0][o1, o1])
               + .5 * (fockb[o2, o2] - focka[o2, o2] - hyb * jmo[1, 1][o2, o2]))
    lr_k = .5 * hyb * (kmo[0, 0][o2, o2] + kmo[1, 1][o1, o1])
    if singlet:
        hd[0] = fockb[o1, o1] - focka[o2, o2] - hyb * jmo[1, 1][o1, o1]
        hd[1] = fockb[o2, o2] - focka[o1, o1] - hyb * jmo[0, 0][o2, o2]
        hd[2] = lr_diag + lr_k
    else:
        hd[0] = lr_diag - lr_k
    if occ_diag is None:
        occ_diag = _occ_diag_jk(mf, nc + 2)
    # the Coulomb term (pp|qq) is large in spin-flip; without it Davidson can miss roots
    hd[noo:] = (fockb[cols[:, 1], cols[:, 1]] - focka[cols[:, 0], cols[:, 0]]
                - hyb * occ_diag[0][cols[:, 0], cols[:, 1]])
    if spc:
        hd[noo:] += cmat.diagonal()

    def vind_mrsf(xm):
        nvec = xm.shape[0]
        xs = numpy.zeros((nvec, (nc + 2) * nvirb))
        if singlet:
            xs[:, i_g] = xm[:, 0]
            xs[:, i_d] = xm[:, 1]
            xs[:, i_11] = xm[:, 2] / SQRT2
            xs[:, i_22] = -xm[:, 2] / SQRT2
        else:
            xs[:, i_11] = xm[:, 0] / SQRT2
            xs[:, i_22] = xm[:, 0] / SQRT2
        xs[:, i456] = xm[:, noo:]
        ys = numpy.asarray(vind_sf(xs)).reshape(nvec, -1)
        ym = numpy.empty_like(xm)
        if singlet:
            ym[:, 0] = ys[:, i_g]
            ym[:, 1] = ys[:, i_d]
            ym[:, 2] = (ys[:, i_11] - ys[:, i_22]) / SQRT2
        else:
            ym[:, 0] = (ys[:, i_11] + ys[:, i_22]) / SQRT2
        ym[:, noo:] = ys[:, i456]
        if spc:
            ym[:, noo:] += xm[:, noo:].dot(cmat)
        return ym
    return vind_mrsf, hd


def _is_cell(mol):
    try:
        from pyscf.pbc import gto as pbcgto
    except ImportError:
        return False
    return isinstance(mol, pbcgto.Cell)


def cv_proxy(mf, mo_coeff, mo_occ, mo_core=None, fock_cv=None):
    '''Closed-shell singlet proxy (O1 doubly occupied, O2 empty) of the triplet reference
    for the EMRSF CV block. Its orbitals are mo_core + mo_coeff, so the proxy
    density, the Fock matrix and (for KS) the XC kernel contain the frozen core. Built
    with the class family of mf (molecule or PBC cell) and the same integral machinery.

    Returns (mf_cv, f_cv, hyb, ncore): f_cv is the proxy Fock matrix in the mo_coeff
    basis (fock_cv if given), ncore the number of frozen core orbitals placed first in
    mf_cv.mo_coeff.'''
    mo_coeff = numpy.asarray(mo_coeff)
    nao = mo_coeff.shape[0]
    mo_core = numpy.zeros((nao, 0)) if mo_core is None else numpy.asarray(mo_core)
    ncore = mo_core.shape[1]
    occ = numpy.asarray(mo_occ, dtype=float)
    o1, o2 = numpy.where(occ == 1)[0]
    occ_s = occ.copy()
    occ_s[o1], occ_s[o2] = 2, 0

    mol0 = mf.mol.copy()
    mol0.spin = 0
    mol0.build(False, False)
    if isinstance(mf, KohnShamDFT):
        mf_cv = mol0.RKS()
        mf_cv.xc = mf.xc
        mf_cv.grids = mf.grids
        hyb = mf_cv._numint.rsh_and_hybrid_coeff(mf_cv.xc, 0)[2]
    else:
        mf_cv = mol0.RHF()
        hyb = 1.
    if _is_cell(mol0):
        mf_cv.with_df = mf.with_df
        mf_cv.exxdiv = mf.exxdiv
    elif getattr(mf, 'with_df', None) is not None:
        mf_cv = mf_cv.density_fit()
        mf_cv.with_df = mf.with_df
    else:
        # reuse the incore AO integrals and keep a direct reference direct
        mf_cv._eri = mf._eri
        mf_cv.direct_scf = mf.direct_scf
        if '_is_mem_enough' in mf.__dict__:
            mf_cv._is_mem_enough = mf._is_mem_enough
    # an SCF object from mf_given_mo_ints carries its integrals as instance attributes
    for key in ('get_hcore', 'get_ovlp', 'energy_nuc'):
        if key in mf.__dict__:
            setattr(mf_cv, key, mf.__dict__[key])
    mf_cv.verbose = 0
    c = numpy.hstack((mo_core, mo_coeff))
    occ_full = numpy.concatenate((numpy.full(ncore, 2.), occ_s))
    mf_cv.mo_coeff = c
    mf_cv.mo_occ = occ_full
    f_full = c.T.dot(mf_cv.get_fock(dm=mf_cv.make_rdm1(c, occ_full))).dot(c)
    f_cv = f_full[ncore:, ncore:] if fock_cv is None else numpy.asarray(fock_cv)
    # TDA of the proxy uses mo_energy for its diagonal; the CV block needs diag(f_cv)
    mf_cv.mo_energy = numpy.concatenate((f_full.diagonal()[:ncore], f_cv.diagonal()))
    return mf_cv, f_cv, hyb, ncore


def gen_ext_vind(mf, hyb, cv, a00, focka, fockb, jmo, kmo, cidx, oidx, vidx, occ_diag=None,
                 singlet=True):
    '''EMRSF extension: CV block (pyscf TDA on the closed-shell proxy, whose frozen-core rows
    carry no amplitude) and the D coupling to the MRSF space, from batched AO J/K.
    cv = cv_proxy(...).'''
    o1, o2 = oidx
    nc, nv = len(cidx), len(vidx)
    ncv = nc * nv
    noo = 3 if singlet else 1
    nm = (nc + 2) * (nv + 2) - 4 + noo
    c = mf.mo_coeff
    cc_, cv_ = c[:, cidx], c[:, vidx]
    co1, co2 = c[:, [o1]], c[:, [o2]]

    mf_cv, f_cv, hyb_cv, ncore = cv
    td_cv = mf_cv.TDA()
    td_cv.singlet = singlet
    td_cv.verbose = 0
    vind_cv, hdiag_cv = td_cv.gen_vind()
    dsh = jmo[0, 0].diagonal() - jmo[1, 1].diagonal()
    sh_ia = (dsh[vidx][None, :] - dsh[cidx][:, None]).ravel()
    diag_cv = a00 - (1 - hyb_cv) * sh_ia
    # off-diagonal F' = F_cv + (1-hyb)[(pq|O2O2) - (pq|O1O1)] 
    fcv = f_cv + (1 - hyb_cv) * (jmo[1, 1] - jmo[0, 0])
    fo_off = fcv[numpy.ix_(cidx, cidx)].copy()
    fv_off = fcv[numpy.ix_(vidx, vidx)].copy()
    fo_off[numpy.diag_indices_from(fo_off)] = 0
    fv_off[numpy.diag_indices_from(fv_off)] = 0
    if occ_diag is None:
        occ_diag = _occ_diag_jk(mf, nc + 2)
    jdiag, kdiag = occ_diag[0][numpy.ix_(cidx, vidx)], occ_diag[1][numpy.ix_(cidx, vidx)]
    # A_ia,ia = e_ia + 2(ia|ia) - hyb (ii|aa) for singlets, e_ia - hyb (ii|aa) for triplets
    hd_cv = (hdiag_cv.reshape(ncore + nc + 1, nv + 1)[ncore:-1, 1:].ravel() + diag_cv
             + ((2 * kdiag if singlet else 0) - hyb_cv * jdiag).ravel())

    j12, k12, k21 = jmo[0, 1], kmo[0, 1], kmo[1, 0]
    row_g = f_cv[numpy.ix_(cidx, vidx)]
    if singlet:
        row_lr = (k12 - 2 * j12)[numpy.ix_(cidx, vidx)] / SQRT2
    else:
        row_lr = k12[numpy.ix_(cidx, vidx)] / SQRT2
    k21_cc = k21[numpy.ix_(cidx, cidx)]
    k12_vv = k12[numpy.ix_(vidx, vidx)]
    j12_o1c = j12[o1, cidx]
    j12_o2v = j12[o2, vidx]
    # Fock-type entries of D (CO1 p=r, O2V q=s): the Fock element adds to the general term
    corr_ov2 = (-1 if singlet else 1) * f_cv[numpy.ix_(cidx, [o1])].T
    corr_co1 = -f_cv[o2, vidx][None, :]

    s_cv = slice(noo, noo + ncv)
    s_ov1 = slice(noo + ncv, noo + ncv + nv)
    s_ov2 = slice(noo + ncv + nv, noo + ncv + 2 * nv)
    s_co1 = slice(noo + ncv + 2 * nv, nm, 2)
    s_co2 = slice(noo + ncv + 2 * nv + 1, nm, 2)

    def xcv_of(xm):
        return xm[:, s_cv].reshape(xm.shape[0], nc, nv)

    def d_triplet(xm, x, xcv, u, w, kx, ku, kw, ym):
        '''Writes D_T x into ym and returns D_T^T xm (S1 + S2 of the SI Tables).'''
        nvec = x.shape[0]
        ym[:, 0] = lib.einsum('rs,nrs->n', row_lr, x)
        ym[:, s_cv] = -(lib.einsum('pr,nrq->npq', k21_cc, x)
                        + lib.einsum('nps,qs->npq', x, k12_vv)).reshape(nvec, -1)
        ym[:, s_ov1] = -lib.einsum('r,nrq->nq', j12_o1c, x)
        ym[:, s_ov2] = kx[:, o1, vidx] + lib.einsum('qr,nrq->nq', corr_ov2, x)
        ym[:, s_co1] = kx[:, cidx, o2] + lib.einsum('ps,nps->np', corr_co1, x)
        ym[:, s_co2] = -lib.einsum('s,nps->np', j12_o2v, x)
        ye = (xm[:, 0, None, None] * row_lr
              - lib.einsum('pr,npq->nrq', k21_cc, xcv) - lib.einsum('nrq,qs->nrs', xcv, k12_vv)
              - lib.einsum('r,nq->nrq', j12_o1c, xm[:, s_ov1])
              + ku[:, cidx][:, :, vidx]
              + lib.einsum('qr,nq->nrq', corr_ov2, u)
              + kw[:, cidx][:, :, vidx]
              + lib.einsum('ps,np->nps', corr_co1, w)
              - lib.einsum('s,np->nps', j12_o2v, xm[:, s_co2]))
        return ye.reshape(nvec, -1)

    def vind_d(xm, xe):
        nvec = xe.shape[0]
        x = xe.reshape(nvec, nc, nv)
        u = xm[:, s_ov2]
        w = xm[:, s_co1]
        dm_x = lib.einsum('pr,nrs,qs->npq', cc_, x, cv_)
        dm_u = lib.einsum('p,nq->npq', co1[:, 0], u.dot(cv_.T))
        dm_w = lib.einsum('np,q->npq', w.dot(cc_.T), co2[:, 0])
        vj, vk = mf.get_jk(mf.mol, numpy.concatenate((dm_x, dm_u, dm_w)), hermi=0)
        vj = lib.einsum('npq,pi,qj->nij', vj, c, c)
        vk = lib.einsum('npq,pi,qj->nij', vk, c, c)
        jx, ju, jw = vj[:nvec], vj[nvec:2 * nvec], vj[2 * nvec:]
        kx, ku, kw = vk[:nvec], vk[nvec:2 * nvec], vk[2 * nvec:]

        ym = numpy.zeros_like(xm)
        if not singlet:
            ye = d_triplet(xm, x, xcv_of(xm), u, w, kx, ku, kw, ym)   # fills ym as well
            return hyb * ym, hyb * ye
        ym[:, 0] = lib.einsum('rs,nrs->n', row_g, x)
        ym[:, 2] = lib.einsum('rs,nrs->n', row_lr, x)
        ym[:, s_cv] = (lib.einsum('pr,nrq->npq', k21_cc, x)
                       - lib.einsum('nps,qs->npq', x, k12_vv)).reshape(nvec, -1)
        ym[:, s_ov1] = lib.einsum('r,nrq->nq', j12_o1c, x)
        ym[:, s_ov2] = (2 * jx[:, o1, vidx] - kx[:, o1, vidx]
                        + lib.einsum('qr,nrq->nq', corr_ov2, x))
        ym[:, s_co1] = (kx[:, cidx, o2] - 2 * jx[:, cidx, o2]
                        + lib.einsum('ps,nps->np', corr_co1, x))
        ym[:, s_co2] = -lib.einsum('s,nps->np', j12_o2v, x)

        xcv = xm[:, s_cv].reshape(nvec, nc, nv)
        ye = (xm[:, 0, None, None] * row_g + xm[:, 2, None, None] * row_lr
              + lib.einsum('pr,npq->nrq', k21_cc, xcv) - lib.einsum('nrq,qs->nrs', xcv, k12_vv)
              + lib.einsum('r,nq->nrq', j12_o1c, xm[:, s_ov1])
              + (2 * ju - ku)[:, cidx][:, :, vidx]
              + lib.einsum('qr,nq->nrq', corr_ov2, u)
              + (kw - 2 * jw)[:, cidx][:, :, vidx]
              + lib.einsum('ps,np->nps', corr_co1, w)
              - lib.einsum('s,np->nps', j12_o2v, xm[:, s_co2]))
        return hyb * ym, hyb * ye.reshape(nvec, -1)

    def vind_cv_ext(xe):
        nvec = xe.shape[0]
        z = numpy.zeros((nvec, ncore + nc + 1, nv + 1))
        z[:, ncore:-1, 1:] = xe.reshape(nvec, nc, nv)
        y = numpy.asarray(vind_cv(z.reshape(nvec, -1))).reshape(nvec, ncore + nc + 1, nv + 1)
        y = y[:, ncore:-1, 1:].reshape(nvec, -1) + xe * diag_cv
        x = xe.reshape(nvec, nc, nv)
        y += (lib.einsum('nia,ab->nib', x, fv_off) - lib.einsum('ij,nja->nia', fo_off, x)).reshape(nvec, -1)
        return y

    return vind_d, vind_cv_ext, hd_cv


def gen_tda_operation(mf, mo_coeff, mo_occ, mo_core=None, hyb=None, extended=True, spc=True,
                      singlet=True, fock=None, fock_cv=None, with_df=False):
    '''Return (vind, hdiag, e_ref) for the MRSF (extended=False) or EMRSF matrix, singlet
    or triplet. fock = (focka, fockb) and fock_cv (mo_coeff basis) replace the Fock
    matrices computed from mf. with_df: density-fitted response integrals.'''
    if hyb is None:
        hyb = hybrid_coeff(mf)
    _check_hyb(mf, hyb, extended, fock)
    mfe, focka, fockb, e_ref = mf_embedded(mf, mo_coeff, mo_occ, mo_core, fock)
    if with_df and getattr(mf, 'with_df', None) is None:
        # response J/K from density fitting; the Fock matrices above stay exact
        if extended and fock_cv is None:
            fock_cv = cv_proxy(mf, mo_coeff, mo_occ, mo_core)[1]
        mf, mfe = mf.density_fit(), mfe.density_fit()
    cidx, oidx, vidx = orb_indices(mo_occ)
    jmo, kmo = _open_jk(mfe, oidx)
    occ_diag = _occ_diag_jk(mfe, len(cidx) + 2)
    vind_mrsf, hd = gen_mrsf_vind(mfe, hyb, spc, focka, fockb, jmo, kmo, cidx, oidx, vidx,
                                  occ_diag, singlet)
    nm = hd.size
    if not extended:
        def vind(xs):
            return vind_mrsf(numpy.asarray(xs).reshape(-1, nm))
        return vind, hd, e_ref

    o1, o2 = oidx
    a00 = fockb[o1, o1] - focka[o2, o2] - hyb * jmo[1, 1][o1, o1]
    cv = cv_proxy(mf, mo_coeff, mo_occ, mo_core, fock_cv)
    vind_d, vind_cv_ext, hd_cv = gen_ext_vind(mfe, hyb, cv, a00, focka, fockb, jmo, kmo, cidx,
                                              oidx, vidx, occ_diag, singlet)

    def vind(xs):
        xs = numpy.asarray(xs).reshape(-1, nm + hd_cv.size)
        xm, xe = xs[:, :nm], xs[:, nm:]
        dym, dye = vind_d(xm, xe)
        return numpy.hstack((vind_mrsf(xm) + dym, vind_cv_ext(xe) + dye))
    return vind, numpy.concatenate((hd, hd_cv)), e_ref



def mrsf_x(x, cidx, oidx, vidx, singlet=True):
    '''MRSF vector -> X[p in core+open, q in open+vir] of the Ms=+1 spin-flip response.
    Singlet: G, D on O2->O1, O1->O2 and L-R as +-1/sqrt(2) on O1->O1, O2->O2;
    triplet: L+R as 1/sqrt(2) on both. Entries of CV_ext (if any) are ignored.'''
    o1, o2 = oidx
    nc, nv = len(cidx), len(vidx)
    xmat = numpy.zeros((nc + 2, nv + 2))
    if singlet:
        xmat[o2, 0] = x[0]
        xmat[o1, 1] = x[1]
        xmat[o1, 0] = x[2] / SQRT2
        xmat[o2, 1] = -x[2] / SQRT2
        k = 3
    else:
        xmat[o1, 0] = xmat[o2, 1] = x[0] / SQRT2
        k = 1
    cols = numpy.vstack((pairs(cidx, vidx), pairs(oidx, vidx), pairs(cidx, oidx)))
    xmat[cols[:, 0], cols[:, 1] - nc] = x[k:k + len(cols)]
    return xmat


def trans_rdm1_mrsf(xi, xj, cidx, oidx, vidx, nmo, singlet=True):
    '''MO-basis transition density T[p,q] = <i|a+_p a_q|j> (spin-summed) between two
    MRSF states, as in OpenQP (get_mrsf_transition_density): occ-occ and vir-vir blocks
    only, with a sqrt(2) factor where exactly one of the two orbitals of a pair is open.'''
    nc = len(cidx)
    nocca, nvirb = nc + 2, len(vidx) + 2
    xmi = mrsf_x(xi, cidx, oidx, vidx, singlet)
    xmj = mrsf_x(xj, cidx, oidx, vidx, singlet)
    s = SQRT2 - 1
    t = numpy.zeros((nmo, nmo))
    open_v = numpy.arange(nvirb) < 2
    vv = xmj.T.dot(xmi)
    vv += s * (open_v[:, None] ^ open_v[None, :]) * xmj[oidx].T.dot(xmi[oidx])
    t[nc:nc + nvirb, nc:nc + nvirb] += vv
    open_o = numpy.arange(nocca) >= nc
    oo = xmi.dot(xmj.T)
    oo += s * (open_o[:, None] ^ open_o[None, :]) * xmi[:, :2].dot(xmj[:, :2].T)
    t[:nocca, :nocca] -= oo
    return t


def trans_rdm1_ext(xi, xj, cidx, oidx, vidx, nmo, singlet=True):
    '''CV_ext contributions to the EMRSF transition density (add to trans_rdm1_mrsf).

    CV_ext (c->v) is a singlet (triplet) single excitation of |core^2 O1^2>; its
    couplings to MRSF are exact one-particle matrix elements: CV_ext-CV_ext, the
    G row (singlet only: G = O2 -> O1 spin flip onto |O1^2>), OV2 (O2->v) and CO1
    (c->O1). The CV_ext norm moves one electron from O2 to O1 relative to the MRSF
    reference density.'''
    o1, o2 = oidx
    nc, nv = len(cidx), len(vidx)
    ncv = nc * nv
    k = 3 if singlet else 1
    nm = len(xi) - ncv
    xei, xej = xi[nm:].reshape(nc, nv), xj[nm:].reshape(nc, nv)
    ov2 = slice(k + ncv + nv, k + ncv + 2 * nv)
    ui, uj = xi[ov2], xj[ov2]
    wi, wj = xi[k + ncv + 2 * nv:nm:2], xj[k + ncv + 2 * nv:nm:2]      # CO1
    t = numpy.zeros((nmo, nmo))
    t[numpy.ix_(vidx, vidx)] += xej.T.dot(xei)
    t[numpy.ix_(cidx, cidx)] -= xei.dot(xej.T)
    if singlet:
        t[numpy.ix_(vidx, cidx)] += SQRT2 * xi[0] * xej.T
        t[numpy.ix_(cidx, vidx)] += SQRT2 * xj[0] * xei
    sign = -1 if singlet else 1
    t[o1, cidx] += sign * xej.dot(ui)
    t[cidx, o1] += sign * xei.dot(uj)
    t[vidx, o2] -= wi.dot(xej)
    t[o2, vidx] -= wj.dot(xei)
    sij = xei.ravel().dot(xej.ravel())
    t[o1, o1] += sij
    t[o2, o2] -= sij
    return t


def basis_labels(cidx, oidx, vidx, extended, singlet=True):
    '''(kind, label) of every basis function, MO indices are 1-based.'''
    o1, o2 = oidx
    lab = lambda i: str(i + MO_BASE)
    if singlet:
        out = [('G', 'G   %s^2' % lab(o1)), ('D', 'D   %s^2' % lab(o2)),
               ('L-R', 'L-R %s->%s' % (lab(o1), lab(o2)))]
    else:
        out = [('L+R', 'L+R %s,%s' % (lab(o1), lab(o2)))]
    out += [('CV', 'CV  %s->%s' % (lab(c), lab(v))) for c in cidx for v in vidx]
    out += [('OV', 'OV  %s->%s' % (lab(k), lab(v))) for k in oidx for v in vidx]
    out += [('CO', 'CO  %s->%s' % (lab(c), lab(k))) for c in cidx for k in oidx]
    if extended:
        out += [('CV_ext', 'CVx %s->%s' % (lab(c), lab(v))) for c in cidx for v in vidx]
    return out


def analyze(tdobj, verbose=None):
    log = logger.new_logger(tdobj, verbose)
    if tdobj.xy is None:
        tdobj.kernel()
    cidx, oidx, vidx = orb_indices(tdobj.mo_occ)
    perm = tdobj._perm
    if perm is not None:    # label with the orbital numbers of the input
        cidx, oidx, vidx = perm[cidx], perm[oidx], perm[vidx]
    labels = basis_labels(cidx, oidx, vidx, tdobj.extended, tdobj.singlet)
    kinds = numpy.array([k for k, _ in labels])
    blocks = ['G', 'D', 'L-R', 'L+R', 'CV', 'OV', 'CO', 'CV_ext']
    e0 = tdobj.e[0]
    for i, (x, _) in enumerate(tdobj.xy):
        w = x**2
        log.note('State %d: %12.5f eV vs T, %12.5f eV vs lowest', i,
                 tdobj.e[i] * HARTREE2EV, (tdobj.e[i] - e0) * HARTREE2EV)
        log.note('    weights ' + '  '.join('%s %.3f' % (b, w[kinds == b].sum()) for b in blocks
                                            if (kinds == b).any()))
        for j in numpy.argsort(-w):
            if w[j] < 0.1:
                break
            log.note('    %-20s %10.5f', labels[j][1], x[j])
    return tdobj


class TDA_MRSF(TDBase):
    '''MRSF (extended=False) or EMRSF TDA on a triplet (M_S = 1) ROHF/ROKS reference.

    Args:
        mf : SCF object (molecule or Gamma-point PBC cell) providing the AO integrals and,
            for Kohn-Sham, the functional and grids; or mf_given_mo_ints (given_mo_ints).
        mo_coeff, mo_occ : orbitals spanning the response space and their occupations
            2, 1, 0 with exactly two singly occupied (default mf.mo_coeff, mf.mo_occ),
            reordered as doubly occupied, singly occupied, empty if needed.
        mo_core : (nao, ncore) frozen doubly occupied orbitals (embedding), or None.
        extended : EMRSF if True.
        fock : (focka, fockb) in the mo_coeff basis, replacing those computed from mf.
        fock_cv : closed-shell proxy Fock matrix of the EMRSF CV block, mo_coeff basis.

    Attributes:
        spc : bool
            Include the spin-pairing coupling (default True).
        with_df : bool
            Density-fit the two-electron integrals of the response; the reference and its
            Fock matrices stay exact. A density-fitted reference always gives a fitted
            response.

    Saved results:
        e : energies relative to the triplet reference (Hartree), S0 included
        e_ref, e_tot : reference energy, and e_ref + e
        xy : (x, 0), x in the basis [G, D, L-R] + CV + OV + CO (+ CV_ext) for singlets and
            [L+R] + CV + OV + CO (+ CV_ext) for triplets
    '''
    # lib.davidson1 tolerance on the energies
    conv_tol = 1e-9
    max_space = 50
    extended = False
    spc = getattr(__config__, 'mrsf_rohf_mrsf_TDA_MRSF_spc', True)
    with_df = False
    # dimensions up to this build the matrix from A x and use eigh
    dense_threshold = 200
    # extra roots solved by Davidson and discarded, so that no low state is missed
    nroots_extra = 3

    _keys = {'mo_coeff', 'mo_occ', 'mo_core', 'fock', 'fock_cv', 'extended', 'spc',
             'with_df', 'dense_threshold', 'nroots_extra', 'max_space', 'e_ref'}

    def __init__(self, mf, mo_coeff=None, mo_occ=None, mo_core=None, extended=None,
                 fock=None, fock_cv=None):
        TDBase.__init__(self, mf)
        self.mo_coeff, self.mo_occ, self._perm = order_orbitals(
            mf.mo_coeff if mo_coeff is None else mo_coeff,
            mf.mo_occ if mo_occ is None else mo_occ)
        self.mo_core = None if mo_core is None else numpy.asarray(mo_core)
        self.fock = fock
        self.fock_cv = fock_cv
        if extended is not None:
            self.extended = extended
        self.e_ref = None
        self._hyb = None
        self._e_ref = None

    @classmethod
    def given_mo_ints(cls, h1e, eri, mo_occ, e_core=0., fock=None, fock_cv=None, hyb=None,
                      extended=None, e_ref=None):
        '''MRSF/EMRSF from integrals already given in the MO basis of the response orbitals:
        h1e (frozen-core potential included), eri (n,n,n,n), mo_occ and the constant e_core
        (nuclear repulsion + frozen-core energy). No SCF object is needed. The response
        kernel is hyb*K (hyb = 1 by default); a hyb other than 1, e.g. with Kohn-Sham fock,
        needs fock and MRSF (extended=False). e_ref fixes the reference energy, otherwise
        the energy of (h1e, eri, e_core).'''
        mf = mf_given_mo_ints(h1e, eri, mo_occ, e_core)
        td = cls(mf, mf.mo_coeff, mf.mo_occ, extended=extended, fock=fock, fock_cv=fock_cv)
        td._hyb = hyb
        td._e_ref = e_ref
        return td

    @property
    def method_name(self):
        return ('EMRSF' if self.extended else 'MRSF') + ('' if self.singlet else '-T')

    @property
    def e_tot(self):
        return self.e_ref + numpy.asarray(self.e)

    def get_hyb(self):
        '''Fraction of exact exchange of mf.xc (1 for HF), or the hyb of given_mo_ints.'''
        return self._hyb if self._hyb is not None else hybrid_coeff(self._scf)

    @property
    def dim(self):
        occ = self.mo_occ
        nc, nv = numpy.count_nonzero(occ == 2), numpy.count_nonzero(occ == 0)
        n = (nc + 2) * (nv + 2) - (1 if self.singlet else 3)
        return n + nc * nv if self.extended else n

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        occ = self.mo_occ
        log.info('\n** %s (%s) **', self.method_name, self.__class__.__name__)
        log.info('response orbitals: %d (doubly occupied %d, open 2, empty %d); frozen core: %d',
                 occ.size, numpy.count_nonzero(occ == 2), numpy.count_nonzero(occ == 0),
                 0 if self.mo_core is None else self.mo_core.shape[1])
        log.info('singlet = %s, spin-pairing coupling = %s, with_df = %s',
                 self.singlet, self.spc, self.with_df)
        log.info('hyb = %g, nstates = %d, conv_tol = %g', self.get_hyb(), self.nstates, self.conv_tol)
        return self

    def check_sanity(self):
        lib.StreamObject.check_sanity(self)
        if self.frozen is not None or self.wfnsym is not None:
            raise NotImplementedError('use mo_coeff / mo_core to choose the response space')
        occ = self.mo_occ
        n = occ.size
        if self.mo_coeff.ndim != 2 or self.mo_coeff.shape[1] != n:
            raise ValueError('mo_coeff must be (nao, n) with n = len(mo_occ)')
        if numpy.count_nonzero(occ == 1) != 2:
            raise ValueError('MRSF needs exactly two singly occupied orbitals')
        if self._scf.mol.spin != 2:
            raise ValueError('MRSF needs a triplet (spin = 2) reference')
        if self.fock is not None and numpy.shape(self.fock) != (2, n, n):
            raise ValueError('fock must be (focka, fockb), each (n, n)')
        if self.fock_cv is not None and numpy.shape(self.fock_cv) != (n, n):
            raise ValueError('fock_cv must be (n, n)')
        if self.with_df and self._scf.mol.natm == 0:
            raise ValueError('with_df needs an atomic basis (not available with given_mo_ints)')
        s = self._scf.get_ovlp()
        c = self.mo_coeff if self.mo_core is None else numpy.hstack((self.mo_core, self.mo_coeff))
        err = abs(c.T.dot(s).dot(c) - numpy.eye(c.shape[1])).max()
        if err > 1e-6:
            logger.warn(self, 'mo_core + mo_coeff are not orthonormal (max error %.2e)', err)
        return self

    def gen_vind(self, mf=None):
        vind, hdiag, self.e_ref = gen_tda_operation(
            self._scf, self.mo_coeff, self.mo_occ, self.mo_core, self.get_hyb(), self.extended,
            self.spc, self.singlet, self.fock, self.fock_cv, self.with_df)
        if self._e_ref is not None:
            self.e_ref = self._e_ref
        return vind, hdiag

    def get_ab(self, mf=None):
        '''Dense A matrix built from A x (no B: MRSF is TDA only).'''
        a = numpy.asarray(self.gen_vind()[0](numpy.eye(self.dim)))
        return .5 * (a + a.T)

    def init_guess(self, nstates=None, hdiag=None):
        if nstates is None:
            nstates = self.nstates
        nstates = min(nstates, hdiag.size)
        # keep every entry degenerate with the nstates-th one, as pyscf does
        e_threshold = numpy.sort(hdiag)[nstates - 1] + 1e-5
        idx = numpy.where(hdiag <= e_threshold)[0]
        x0 = numpy.zeros((len(idx), hdiag.size))
        x0[numpy.arange(len(idx)), idx] = 1
        return x0

    def kernel(self, x0=None, nstates=None):
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        log = logger.new_logger(self)
        if nstates is not None:
            self.nstates = nstates
        dim = self.dim
        nroots = min(self.nstates, dim)

        vind, hdiag = self.gen_vind()
        if dim <= self.dense_threshold or nroots >= dim:
            a = numpy.asarray(vind(numpy.eye(dim)))
            w, v = numpy.linalg.eigh(.5 * (a + a.T))
            self.e = w[:nroots]
            xs = v[:, :nroots].T
            self.converged = numpy.ones(nroots, dtype=bool)
        else:
            nsolve = min(nroots + self.nroots_extra, dim)
            if x0 is None:
                x0 = self.init_guess(nsolve, hdiag)
            conv, e, xs = lib.davidson1(
                vind, x0, self.get_precond(hdiag), tol=self.conv_tol, max_cycle=self.max_cycle,
                max_space=self.max_space, lindep=self.lindep, nroots=nsolve, verbose=log)
            idx = numpy.argsort(e)[:nroots]
            self.converged = numpy.asarray(conv)[idx]
            self.e = numpy.asarray(e)[idx]
            xs = [xs[i] for i in idx]
        self.xy = [(x, 0) for x in xs]
        log.timer(self.method_name, *cpu0)
        self._finalize()
        return self.e, self.xy

    def _finalize(self):
        log = logger.new_logger(self)
        if not all(self.converged):
            log.warn('%s states not converged: %s', self.method_name,
                     [i for i, c in enumerate(self.converged) if not c])
        log.note('%s reference energy %.12f; energies relative to it (eV):',
                 self.method_name, self.e_ref)
        for i, e in enumerate(self.e):
            log.note('  state %d  %12.6f', i, e * HARTREE2EV)
        return self

    analyze = analyze

    def trans_rdm1(self, i, j):
        '''Transition density T[p,q] = <i|a+_p a_q|j> (spin-summed) between states i and j
        in the basis of mo_coeff, given in the orbital order of the input.'''
        if self.xy is None:
            self.kernel()
        cidx, oidx, vidx = orb_indices(self.mo_occ)
        n = self.mo_occ.size
        xi, xj = self.xy[i][0], self.xy[j][0]
        t = trans_rdm1_mrsf(xi, xj, cidx, oidx, vidx, n, self.singlet)
        if self.extended:
            t += trans_rdm1_ext(xi, xj, cidx, oidx, vidx, n, self.singlet)
        if self._perm is not None:
            t1 = numpy.empty_like(t)
            t1[numpy.ix_(self._perm, self._perm)] = t
            t = t1
        return t

    def transition_dipole(self):
        '''Transition dipoles (a.u., origin at the center of mass) from the lowest state to
        states 1, 2, ...; shape (nstates-1, 3). Molecules only.'''
        if self.xy is None:
            self.kernel()
        mol = self.mol
        if getattr(mol, 'a', None) is not None or mol.natm == 0:
            raise NotImplementedError('transition dipoles are available for molecules only')
        mass = mol.atom_mass_list()
        com = numpy.einsum('i,ix->x', mass, mol.atom_coords()) / mass.sum()
        with mol.with_common_orig(com):
            ints = mol.intor_symmetric('int1e_r', comp=3)
        c = self.mo_coeff if self._perm is None else self.mo_coeff[:, numpy.argsort(self._perm)]
        dip = [-numpy.einsum('xpq,pq->x', ints, c.dot(self.trans_rdm1(0, i)).dot(c.T))
               for i in range(1, len(self.xy))]
        return numpy.array(dip).reshape(-1, 3)

    def oscillator_strength(self):
        '''Oscillator strengths from the lowest state to states 1, 2, ....'''
        dip = self.transition_dipole()
        return 2. / 3. * (self.e[1:] - self.e[0]) * numpy.einsum('ix,ix->i', dip, dip)

    def _no_transition_property(self, *args, **kwargs):
        raise NotImplementedError('only transition dipoles and oscillator strengths are implemented')

    transition_quadrupole = transition_octupole = \
        transition_velocity_dipole = transition_velocity_quadrupole = \
        transition_velocity_octupole = transition_magnetic_dipole = \
        transition_magnetic_quadrupole = get_nto = _no_transition_property

    def nuc_grad_method(self):
        raise NotImplementedError('MRSF/EMRSF gradients are not implemented')


scf.rohf.ROHF.TDA_MRSF = lib.class_as_method(TDA_MRSF)
