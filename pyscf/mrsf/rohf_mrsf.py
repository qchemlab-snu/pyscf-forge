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
#
'''
Mixed-reference spin-flip (MRSF) and extended MRSF (EMRSF) TDA for a triplet
ROHF/ROKS reference. EMRSF adds the core->virtual configurations of the
closed-shell singlet reference (LR-TDDFT) to the MRSF response space.

Ref: M. Oh, N. Kim, Y. Jung, C. H. Choi, S. Lee, J. Chem. Theory Comput. 22, 6057 (2026)
'''

import numpy
from pyscf import lib, ao2mo, dft, scf, tdscf
from pyscf.lib import logger
from pyscf.data.nist import HARTREE2EV
from pyscf.tdscf.rhf import TDBase
from pyscf import __config__
from pyscf.dft.rks import KohnShamDFT

SQRT2 = numpy.sqrt(2.)


def orb_indices(mf):
    '''Core, open (O1, O2) and virtual MO indices of a triplet ROHF/ROKS reference.'''
    if mf.mo_occ is None:
        raise ValueError('run the SCF first')
    occ = numpy.asarray(mf.mo_occ)
    if occ.ndim != 1:
        raise ValueError('EMRSF needs a ROHF/ROKS reference')
    if numpy.count_nonzero(occ == 1) != 2 or mf.mol.spin != 2:
        raise ValueError('EMRSF needs a triplet reference with exactly two singly occupied orbitals')
    if numpy.any(numpy.diff(occ) > 0):
        raise ValueError('mo_occ must be ordered as core, open, virtual')
    nc = numpy.count_nonzero(occ == 2)
    return (numpy.arange(nc), numpy.arange(nc, nc + 2), numpy.arange(nc + 2, occ.size))


def pairs(a, b):
    return numpy.array([(p, q) for p in a for q in b], dtype=int).reshape(-1, 2)


def mo_fock(mf):
    '''alpha and beta Fock matrices of the ROHF/ROKS reference in the MO basis.'''
    f = mf.get_fock(dm=mf.make_rdm1())
    c = mf.mo_coeff
    return c.T.dot(f.focka).dot(c), c.T.dot(f.fockb).dot(c)


def mo_eri(mf):
    nmo = mf.mo_coeff.shape[1]
    return ao2mo.full(mf.mol, mf.mo_coeff, compact=False).reshape(nmo, nmo, nmo, nmo)


def gen_block(focka, fockb, eri, hyb, rows, cols):
    '''A[(p,q),(r,s)] = d_pr Fb[q,s] - d_qs Fa[p,r] - hyb (pr|qs)'''
    p, q = rows[:, 0, None], rows[:, 1, None]
    r, s = cols[None, :, 0], cols[None, :, 1]
    a = -hyb * eri[p, r, q, s]
    a += (p == r) * fockb[q, s]
    a -= (q == s) * focka[p, r]
    return a


def spin_pair_coupling(eri, hyb, cidx, oidx, vidx):
    '''Spin-pairing coupling between the Ms=+1 and Ms=-1 responses in the
    CV + OV + CO basis. CV rows and columns are zero.'''
    o1, o2 = oidx
    nc, nv = len(cidx), len(vidx)
    ncv = nc * nv
    n = ncv + 2 * nv + 2 * nc
    vr, vc = vidx[:, None], vidx[None, :]
    cr, cc = cidx[:, None], cidx[None, :]
    ov = [slice(ncv + k * nv, ncv + (k + 1) * nv) for k in range(2)]
    co = [slice(ncv + 2 * nv + k, n, 2) for k in range(2)]
    opn = (o1, o2)
    cmat = numpy.zeros((n, n))
    for k in range(2):
        ok, kp = opn[k], opn[1 - k]
        cmat[ov[k], ov[k]] = -eri[kp, vc, vr, kp]
        cmat[co[k], co[k]] = -eri[cc, kp, kp, cr]
        cmat[ov[k], ov[1 - k]] = eri[kp, vc, vr, ok]
        cmat[co[k], co[1 - k]] = eri[cr, ok, cc, kp]
    x = eri[vr, o1, cc, o2] - eri[vr, o2, cc, o1]
    cmat[ov[0], co[1]] = -x
    cmat[ov[1], co[0]] = x
    cmat[co[1], ov[0]] = -x.T
    cmat[co[0], ov[1]] = x.T
    return hyb * cmat


def mrsf_singlet(focka, fockb, eri, hyb, cidx, oidx, vidx, spc=True):
    '''MRSF singlet matrix in the basis [G, D, L-R] + CV + OV + CO.'''
    o1, o2 = oidx
    a3 = numpy.empty((3, 3))
    a3[0, 0] = fockb[o1, o1] - focka[o2, o2] - hyb * eri[o2, o2, o1, o1]
    a3[0, 1] = a3[1, 0] = -hyb * eri[o2, o1, o1, o2]
    a3[1, 1] = fockb[o2, o2] - focka[o1, o1] - hyb * eri[o1, o1, o2, o2]
    a3[0, 2] = a3[2, 0] = ((-focka[o2, o1] - hyb * eri[o2, o1, o1, o1])
                           - (fockb[o1, o2] - hyb * eri[o2, o2, o1, o2])) / SQRT2
    a3[1, 2] = a3[2, 1] = ((fockb[o2, o1] - hyb * eri[o1, o1, o2, o1])
                           - (-focka[o1, o2] - hyb * eri[o1, o2, o2, o2])) / SQRT2
    a3[2, 2] = (.5 * (fockb[o1, o1] - focka[o1, o1] - hyb * eri[o1, o1, o1, o1])
                + .5 * (fockb[o2, o2] - focka[o2, o2] - hyb * eri[o2, o2, o2, o2])
                + .5 * hyb * (eri[o1, o2, o1, o2] + eri[o2, o1, o2, o1]))

    cols = numpy.vstack((pairs(cidx, vidx), pairs(oidx, vidx), pairs(cidx, oidx)))
    a_gd = gen_block(focka, fockb, eri, hyb, numpy.array([[o2, o1], [o1, o2]]), cols)
    a_lr = (gen_block(focka, fockb, eri, hyb, numpy.array([[o1, o1]]), cols)
            - gen_block(focka, fockb, eri, hyb, numpy.array([[o2, o2]]), cols)) / SQRT2
    a_oo_x = numpy.vstack((a_gd, a_lr))
    a_xx = gen_block(focka, fockb, eri, hyb, cols, cols)
    if spc:
        a_xx -= spin_pair_coupling(eri, hyb, cidx, oidx, vidx)
    return numpy.block([[a3, a_oo_x], [a_oo_x.T, a_xx]])


def mrsf_triplet(focka, fockb, eri, hyb, cidx, oidx, vidx, spc=True):
    '''MRSF triplet matrix (Ms=0 components) in the basis [L+R] + CV + OV + CO.'''
    o1, o2 = oidx
    a_lr = (.5 * (fockb[o1, o1] - focka[o1, o1] - hyb * eri[o1, o1, o1, o1])
            + .5 * (fockb[o2, o2] - focka[o2, o2] - hyb * eri[o2, o2, o2, o2])
            - .5 * hyb * (eri[o1, o2, o1, o2] + eri[o2, o1, o2, o1]))
    cols = numpy.vstack((pairs(cidx, vidx), pairs(oidx, vidx), pairs(cidx, oidx)))
    a_lr_x = (gen_block(focka, fockb, eri, hyb, numpy.array([[o1, o1]]), cols)
              + gen_block(focka, fockb, eri, hyb, numpy.array([[o2, o2]]), cols)) / SQRT2
    a_xx = gen_block(focka, fockb, eri, hyb, cols, cols)
    if spc:
        a_xx += spin_pair_coupling(eri, hyb, cidx, oidx, vidx)
    return numpy.block([[numpy.array([[a_lr]]), a_lr_x], [a_lr_x.T, a_xx]])


def cv_proxy(mf, xc_cv=None):
    '''Closed-shell singlet proxy (O1 doubly occupied, O2 empty) of the triplet
    reference, its MO Fock matrix and hybrid coefficient.'''
    cidx, oidx, vidx = orb_indices(mf)
    c = mf.mo_coeff
    o1, o2 = oidx
    occ_s = numpy.array(mf.mo_occ, dtype=float)
    occ_s[o1], occ_s[o2] = 2, 0
    # the proxy must be a spin=0 molecule, otherwise get_ab evaluates the kernel wrongly
    mol0 = mf.mol.copy()
    mol0.spin = 0
    mol0.build(False, False)
    if isinstance(mf, KohnShamDFT):
        mf_cv = dft.RKS(mol0)
        mf_cv.xc = xc_cv or mf.xc
        mf_cv.grids = mf.grids
        hyb = mf_cv._numint.rsh_and_hybrid_coeff(mf_cv.xc, 0)[2]
    else:
        if xc_cv:
            raise ValueError('xc_cv needs a Kohn-Sham (ROKS) reference')
        mf_cv = scf.RHF(mol0)
        hyb = 1.
    if getattr(mf, 'with_df', None) is not None:
        mf_cv = mf_cv.density_fit()
        mf_cv.with_df = mf.with_df
    else:
        # reuse the incore AO integrals and keep a direct reference direct, like mf.to_uks()
        mf_cv._eri = mf._eri
        mf_cv.direct_scf = mf.direct_scf
        if '_is_mem_enough' in mf.__dict__:
            mf_cv._is_mem_enough = mf._is_mem_enough
    mf_cv.verbose = 0
    mf_cv.mo_coeff = c
    mf_cv.mo_occ = occ_s
    f_cv = c.T.dot(mf_cv.get_fock(dm=mf_cv.make_rdm1(c, occ_s))).dot(c)
    return mf_cv, f_cv, hyb


def cv_offdiag_fock(f_cv, eri, hyb, cidx, oidx, vidx):
    '''Off-diagonal core-core and virtual-virtual blocks of the CV-block Fock matrix
    F' = F_cv + (1-hyb)[(pq|O2O2) - (pq|O1O1)] (paper eq 16, taken as a matrix).'''
    o1, o2 = oidx
    f = f_cv + (1 - hyb) * (eri[:, :, o2, o2] - eri[:, :, o1, o1])
    fo = f[numpy.ix_(cidx, cidx)].copy()
    fv = f[numpy.ix_(vidx, vidx)].copy()
    fo[numpy.diag_indices_from(fo)] = 0
    fv[numpy.diag_indices_from(fv)] = 0
    return fo, fv


def cv_block(mf, xc_cv, eri, cidx, oidx, vidx, singlet=True):
    '''LR-TDA A matrix (singlet or triplet excitations) of the closed-shell singlet
    reference (O1 doubly occupied, O2 empty) restricted to core -> virtual
    excitations, and the Fock matrix of that reference.

    The one-particle part is the full matrix F' of paper eqs 11, 12 and 16 (off-diagonal
    elements included), so the block is invariant to rotations among core and among
    virtual orbitals; the published implementation kept only diag(F').'''
    o1, o2 = oidx
    nc, nv = len(cidx), len(vidx)
    mf_cv, f_cv, hyb = cv_proxy(mf, xc_cv)
    nocc, nvir = nc + 1, nv + 1
    if singlet:
        a = tdscf.rhf.get_ab(mf_cv, mo_energy=f_cv.diagonal().copy(), mo_coeff=mf_cv.mo_coeff,
                             mo_occ=mf_cv.mo_occ)[0]
    else:
        # pyscf get_ab has no triplet option; build it from the triplet TDA product
        mf_cv.mo_energy = f_cv.diagonal().copy()
        td = tdscf.rhf.TDA(mf_cv)
        td.singlet = False
        td.verbose = 0
        vind = td.gen_vind()[0]
        a = numpy.asarray(vind(numpy.eye(nocc * nvir))).reshape(nocc, nvir, nocc, nvir)
    sh = numpy.einsum('xx->x', eri[:, :, o1, o1]) - numpy.einsum('xx->x', eri[:, :, o2, o2])
    sh_ia = sh[nocc:][None, :] - sh[:nocc][:, None]
    a = a.reshape(nocc * nvir, nocc * nvir)
    a[numpy.diag_indices_from(a)] -= sh_ia.ravel() * (1 - hyb)
    a = a.reshape(nocc, nvir, nocc, nvir)[:-1, 1:, :-1, 1:].reshape(nc * nv, nc * nv)
    fo, fv = cv_offdiag_fock(f_cv, eri, hyb, cidx, oidx, vidx)
    a += numpy.kron(numpy.eye(nc), fv) - numpy.kron(fo, numpy.eye(nv))
    return a, f_cv


def d_block(eri, f_cv, hyb, cidx, oidx, vidx):
    '''Coupling between the MRSF space ([G, D, L-R] + CV + OV + CO) and the
    CV extension.

    The CV/OV/CO rows are S1 - S2 of the paper's SI Tables S1 (Ms=+1) and S2 (Ms=-1),
    with the Slater-Condon value of their Fock-type entries: for CO1 (p=r) and O2V
    (q=s) the Fock element of the closed-shell O1^2 reference is added to the general
    expression, because the two determinants share Cp (Vq) singly occupied. The
    tables give the Fock element alone, which is not invariant to rotations among
    core or virtual orbitals.'''
    o1, o2 = oidx
    cv, ov, co = pairs(cidx, vidx), pairs(oidx, vidx), pairs(cidx, oidx)
    r, s = cv[None, :, 0], cv[None, :, 1]

    d_oo = numpy.zeros((3, len(cv)))
    d_oo[0] = f_cv[cv[:, 0], cv[:, 1]]
    d_oo[2] = (eri[o1, cv[:, 0], o2, cv[:, 1]] - 2 * eri[o1, o2, cv[:, 0], cv[:, 1]]) / SQRT2

    p, q = cv[:, 0, None], cv[:, 1, None]
    d_cv = (q == s) * eri[p, o2, o1, r] - (p == r) * eri[o1, q, o2, s]

    p, q = ov[:, 0, None], ov[:, 1, None]
    from_o2 = 2 * eri[o1, q, r, s] - eri[o1, r, q, s] - (q == s) * f_cv[r, o1]
    d_ov = numpy.where(p == o1, (q == s) * eri[o1, o2, o1, r], from_o2)

    p, q = co[:, 0, None], co[:, 1, None]
    to_o1 = eri[p, r, o2, s] - 2 * eri[p, o2, r, s] - (p == r) * f_cv[o2, s]
    d_co = numpy.where(q == o1, to_o1, (p == r) * (-eri[o1, o2, o2, s]))

    return hyb * numpy.vstack((d_oo, d_cv, d_ov, d_co))


def d_block_triplet(eri, f_cv, hyb, cidx, oidx, vidx):
    '''Coupling between the triplet MRSF space ([L+R] + CV + OV + CO) and the
    triplet CV extension: S1 + S2 of the paper's SI Tables (Fock-type entries as in
    d_block), and (O1 r|O2 s)/sqrt(2) for L+R, the sqrt(2) convention of the singlet
    L-R row.'''
    o1, o2 = oidx
    cv, ov, co = pairs(cidx, vidx), pairs(oidx, vidx), pairs(cidx, oidx)
    r, s = cv[None, :, 0], cv[None, :, 1]

    d_lr = eri[o1, cv[:, 0], o2, cv[:, 1]][None, :] / SQRT2

    p, q = cv[:, 0, None], cv[:, 1, None]
    d_cv = (p == r) * (-eri[o1, q, o2, s]) - (q == s) * eri[p, o2, o1, r]

    p, q = ov[:, 0, None], ov[:, 1, None]
    from_o2 = eri[o1, r, q, s] + (q == s) * f_cv[r, o1]
    d_ov = numpy.where(p == o1, (q == s) * (-eri[o1, o2, o1, r]), from_o2)

    p, q = co[:, 0, None], co[:, 1, None]
    to_o1 = eri[p, r, o2, s] - (p == r) * f_cv[o2, s]
    d_co = numpy.where(q == o1, to_o1, (p == r) * (-eri[o1, o2, o2, s]))

    return hyb * numpy.vstack((d_lr, d_cv, d_ov, d_co))


def g_energy(focka, fockb, eri, hyb, oidx):
    '''Energy of the closed-shell singlet O1^2 relative to the triplet reference in
    the MRSF blocks: the [G, G] element, shifted onto the CV extension.'''
    o1, o2 = oidx
    return fockb[o1, o1] - focka[o2, o2] - hyb * eri[o2, o2, o1, o1]


def build_matrix(mf, hyb, expansion=True, spc=True, xc_cv=None, singlet=True):
    cidx, oidx, vidx = orb_indices(mf)
    focka, fockb = mo_fock(mf)
    eri = mo_eri(mf)
    if singlet:
        a = mrsf_singlet(focka, fockb, eri, hyb, cidx, oidx, vidx, spc)
    else:
        a = mrsf_triplet(focka, fockb, eri, hyb, cidx, oidx, vidx, spc)
    if not expansion:
        return a
    a_cv, f_cv = cv_block(mf, xc_cv, eri, cidx, oidx, vidx, singlet)
    a_cv[numpy.diag_indices_from(a_cv)] += g_energy(focka, fockb, eri, hyb, oidx)
    if singlet:
        d = d_block(eri, f_cv, hyb, cidx, oidx, vidx)
    else:
        d = d_block_triplet(eri, f_cv, hyb, cidx, oidx, vidx)
    return numpy.block([[a, d], [d.T, a_cv]])


def hybrid_coeff(mf):
    '''Fraction of exact exchange in mf.xc (1 for HF).'''
    if isinstance(mf, KohnShamDFT):
        return mf._numint.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)[2]
    return 1.


def get_ab_mrsf(mf, hyb=None, expansion=True, spc=True, xc_cv=None, singlet=True):
    '''Dense MRSF (expansion=False) or EMRSF singlet matrix A in the basis
    [G, D, L-R] + CV + OV + CO (+ CV_ext).

    MRSF is defined in the Tamm-Dancoff form only, so a single 2-D matrix A is
    returned. pyscf get_ab and sftda get_ab_sf return (A, B) because they also
    serve full TDDFT; there is no B here. hyb defaults to hybrid_coeff(mf).'''
    if hyb is None:
        hyb = hybrid_coeff(mf)
    return build_matrix(mf, hyb, expansion, spc, xc_cv, singlet)


get_ab = get_ab_mrsf

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


MO_BASE = getattr(__config__, 'MO_BASE', 1)


def basis_labels(cidx, oidx, vidx, expansion, singlet=True):
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
    if expansion:
        out += [('CV_ext', 'CVx %s->%s' % (lab(c), lab(v))) for c in cidx for v in vidx]
    return out


def analyze(tdobj, verbose=None):
    log = logger.new_logger(tdobj, verbose)
    if tdobj.xy is None:
        tdobj.kernel()
    cidx, oidx, vidx = orb_indices(tdobj._scf)
    labels = basis_labels(cidx, oidx, vidx, tdobj.expansion, tdobj.singlet)
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
    '''MRSF singlet TDA on a triplet ROHF/ROKS reference (TDA_EMRSF adds CV_ext).

    Attributes:
        e : energies relative to the triplet reference (Hartree), S0 included
        xy : list of (x, 0), x in the basis [G, D, L-R] + CV + OV + CO (+ CV_ext)
        dense_threshold : dimensions up to this use the dense matrix and eigh
        nroots_extra : extra roots solved by Davidson (and discarded) so that no low state
            is missed
    '''
    # lib.davidson1 tolerance on the energies (TDBase's default is a residual tolerance)
    conv_tol = 1e-9
    max_space = 50
    expansion = False
    spc = True
    hyb = None
    xc_cv = None
    dense_threshold = 200
    nroots_extra = 3

    _keys = {'expansion', 'spc', 'hyb', 'xc_cv', 'dense_threshold', 'nroots_extra', 'max_space',
             'max_memory'}

    def __init__(self, mf, expansion=None):
        TDBase.__init__(self, mf)
        if expansion is not None:
            self.expansion = expansion
        self._a = None
        self._a_key = None
        self._a_src = (None, None, None)

    @property
    def method_name(self):
        return ('EMRSF' if self.expansion else 'MRSF') + ('' if self.singlet else '-T')

    def get_hyb(self):
        '''Fraction of exact exchange: td.hyb if set, else that of mf.xc.'''
        if self.hyb is not None:
            return self.hyb
        return hybrid_coeff(self._scf)

    # names used before the pyscf-style hyb
    get_hfx = get_hyb

    @property
    def hfx(self):
        return self.hyb

    @hfx.setter
    def hfx(self, x):
        self.hyb = x

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n** %s (%s) **', self.method_name, self.__class__.__name__)
        log.info('expansion (EMRSF) = %s', self.expansion)
        log.info('singlet = %s', self.singlet)
        log.info('spin-pairing coupling = %s', self.spc)
        log.info('hyb = %g', self.get_hyb())
        log.info('xc for CV block = %s', self.xc_cv or getattr(self._scf, 'xc', 'HF'))
        log.info('nstates = %d', self.nstates)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('dense_threshold = %d', self.dense_threshold)
        return self

    def check_sanity(self):
        lib.StreamObject.check_sanity(self)
        if self.frozen is not None or self.wfnsym is not None:
            raise NotImplementedError('MRSF/EMRSF does not support frozen orbitals or wfnsym')
        orb_indices(self._scf)
        if not self._scf.converged:
            logger.warn(self, 'Reference SCF is not converged')
        return self

    def reset(self, mol=None):
        TDBase.reset(self, mol)
        self._a = None
        self._a_key = None
        self._a_src = (None, None, None)
        return self

    def get_ab_mrsf(self):
        '''Dense A matrix of this object (no B: MRSF is TDA only), cached.'''
        mf = self._scf
        key = (self.expansion, self.spc, self.get_hyb(), self.xc_cv, getattr(mf, 'xc', None),
               self.singlet)
        # a rerun SCF replaces mo_coeff/mo_occ, which must invalidate the cached matrix
        src = (mf, mf.mo_coeff, mf.mo_occ)
        if (self._a is None or self._a_key != key
                or any(a is not b for a, b in zip(self._a_src, src))):
            if getattr(mf, 'with_df', None) is not None:
                # build from the same fitted operator as Davidson, so DF results do not
                # depend on dense_threshold
                self._a = numpy.asarray(self.gen_vind()[0](numpy.eye(self.dim)))
            else:
                self._a = build_matrix(mf, self.get_hyb(), self.expansion, self.spc, self.xc_cv,
                                       self.singlet)
            self._a_key = key
            self._a_src = src
        return self._a

    @property
    def dim(self):
        cidx, oidx, vidx = orb_indices(self._scf)
        nc, nv = len(cidx), len(vidx)
        # singlet: G, D, L-R from the four open-open pairs; triplet: L+R only
        n = (nc + 2) * (nv + 2) - (1 if self.singlet else 3)
        if self.expansion:
            n += nc * nv
        return n

    def gen_vind(self):
        from pyscf.mrsf.mrsf_hop import gen_tda_operation
        return gen_tda_operation(self._scf, self.get_hyb(), self.expansion, self.spc, self.xc_cv,
                                 self.singlet)

    def get_hdiag(self):
        if self.dim <= self.dense_threshold:
            return self.get_ab_mrsf().diagonal().copy()
        return self.gen_vind()[1]

    def init_guess(self, nstates=None, hdiag=None):
        if nstates is None:
            nstates = self.nstates
        if hdiag is None:
            hdiag = self.get_hdiag()
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

        if dim <= self.dense_threshold or nroots >= dim:
            w, v = numpy.linalg.eigh(self.get_ab_mrsf())
            self.e = w[:nroots]
            xs = v[:, :nroots].T
            self.converged = numpy.ones(nroots, dtype=bool)
        else:
            vind, hdiag = self.gen_vind()
            # a few extra roots: davidson1 stops once the requested roots converge, and a
            # lower state that is weakly coupled to the guess space can be missed
            nsolve = min(nroots + self.nroots_extra, dim)
            if x0 is None:
                x0 = self.init_guess(nsolve, hdiag)

            # lib.davidson1 instead of tdscf._lr_eig.eigh: the latter adds up to 20 trial
            # vectors per iteration and needed 2-4x more A x products here
            conv, e, xs = lib.davidson1(
                vind, x0, self.get_precond(hdiag), tol=self.conv_tol, max_cycle=self.max_cycle,
                max_space=self.max_space, lindep=self.lindep, nroots=nsolve, verbose=log)
            idx = numpy.argsort(e)[:nroots]
            self.converged = numpy.asarray(conv)[idx]
            self.e = numpy.asarray(e)[idx]
            xs = [xs[i] for i in idx]
        self.xy = [(x, 0) for x in xs]
        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)
        log.timer(self.method_name, *cpu0)
        self._finalize()
        return self.e, self.xy

    def _finalize(self):
        log = logger.new_logger(self)
        if not all(self.converged):
            log.warn('%s states not converged: %s', self.method_name,
                     [i for i, c in enumerate(self.converged) if not c])
        log.note('%s energies relative to the triplet reference (eV):', self.method_name)
        for i, e in enumerate(self.e):
            log.note('  state %d  %12.6f', i, e * HARTREE2EV)
        return self

    # name used before get_ab_mrsf; note it differs from TDBase.get_ab(mf, frozen)
    get_ab = get_ab_mrsf

    analyze = analyze

    def nuc_grad_method(self):
        raise NotImplementedError('MRSF/EMRSF gradients are not implemented')

    def _check_transition_properties(self):
        if self.xy is None:
            self.kernel()

    def trans_rdm1(self, i, j):
        '''MO-basis transition density T[p,q] = <i|a+_p a_q|j> (spin-summed) between
        states i and j.'''
        self._check_transition_properties()
        cidx, oidx, vidx = orb_indices(self._scf)
        nmo = self._scf.mo_coeff.shape[1]
        xi, xj = self.xy[i][0], self.xy[j][0]
        t = trans_rdm1_mrsf(xi, xj, cidx, oidx, vidx, nmo, self.singlet)
        if self.expansion:
            t += trans_rdm1_ext(xi, xj, cidx, oidx, vidx, nmo, self.singlet)
        return t

    def transition_dipole(self):
        '''Transition dipoles (a.u., origin at the center of mass) from the lowest state
        to states 1, 2, ...; shape (nstates-1, 3).'''
        self._check_transition_properties()
        mol = self.mol
        mass = mol.atom_mass_list()
        com = numpy.einsum('i,ix->x', mass, mol.atom_coords()) / mass.sum()
        with mol.with_common_orig(com):
            ints = mol.intor_symmetric('int1e_r', comp=3)
        c = self._scf.mo_coeff
        dip = [-numpy.einsum('xpq,pq->x', ints, c.dot(self.trans_rdm1(0, i)).dot(c.T))
               for i in range(1, len(self.xy))]
        return numpy.array(dip).reshape(-1, 3)

    def oscillator_strength(self):
        '''Oscillator strengths from the lowest state to states 1, 2, ....'''
        dip = self.transition_dipole()
        return 2. / 3. * (self.e[1:] - self.e[0]) * numpy.einsum('ix,ix->i', dip, dip)

    def _no_transition_property(self, *args, **kwargs):
        raise NotImplementedError('transition properties and NTOs are not implemented for MRSF/EMRSF')

    # the TDBase versions assume x of shape (nocc, nvir) and fail with obscure errors;
    # only the dipole and oscillator strength are implemented (above)
    transition_quadrupole = transition_octupole = \
        transition_velocity_dipole = transition_velocity_quadrupole = \
        transition_velocity_octupole = transition_magnetic_dipole = \
        transition_magnetic_quadrupole = get_nto = _no_transition_property


class TDA_EMRSF(TDA_MRSF):
    expansion = True


EMRSF_TDA = TDA = TDA_EMRSF

scf.rohf.ROHF.TDA_MRSF = lib.class_as_method(TDA_MRSF)
scf.rohf.ROHF.TDA_EMRSF = lib.class_as_method(TDA_EMRSF)
