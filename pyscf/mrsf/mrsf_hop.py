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
Matrix-free A x for the MRSF / EMRSF singlet TDA.

The MRSF generic part reuses the collinear spin-flip TDA of pyscf-forge
(pyscf.sftda), the CV extension reuses pyscf.tdscf.rhf.TDA on a closed-shell
proxy, and the remaining couplings are built from AO J/K contractions.
'''

import numpy
from pyscf import lib, ao2mo, tdscf
from pyscf.dft.rks import KohnShamDFT
from pyscf.mrsf.rohf_mrsf import SQRT2, orb_indices, ordered_scf, pairs, mo_fock, cv_proxy


def _check_supported(mf, hyb):
    if isinstance(mf, KohnShamDFT):
        omega, alpha, hyb_xc = mf._numint.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)
        if omega != 0:
            raise NotImplementedError('matrix-free EMRSF does not support range-separated functionals')
    else:
        hyb_xc = 1.
    if abs(hyb - hyb_xc) > 1e-12:
        raise NotImplementedError('matrix-free EMRSF needs hyb equal to the hybrid coefficient of mf.xc')


def _sf_vind(mf):
    try:
        from pyscf.sftda.uhf_sf import TDA_SF
    except ImportError:
        raise ImportError('matrix-free EMRSF needs pyscf.sftda from pyscf-forge')
    mfu = mf.to_uks() if isinstance(mf, KohnShamDFT) else mf.to_uhf()
    td = TDA_SF(mfu, extype=1, collinear_samples=-1)
    td.verbose = 0
    return td.gen_vind()[0]


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


def _occ_diag(mf, nocc):
    '''(pp|xx) and (px|px) in the MO basis for the alpha-occupied orbitals p (core + open).

    Built from density-fitting three-index tensors in O(N^4) time and O(N^3) memory. A
    reference without DF gets a temporary DF object; these integrals only enter hdiag
    (initial guess and preconditioner), so the fitting error does not reach the energies.'''
    from pyscf import df
    mydf = getattr(mf, 'with_df', None)
    if mydf is None:
        mydf = df.DF(mf.mol)
        mydf.max_memory = mf.max_memory
    c = mf.mo_coeff
    nmo = c.shape[1]
    co = c[:, :nocc]
    diag = numpy.arange(nocc)
    jdiag = numpy.zeros((nocc, nmo))
    kdiag = numpy.zeros((nocc, nmo))
    for eri1 in mydf.loop():
        lpq = lib.unpack_tril(eri1)
        lmx = lib.einsum('Lmn,nx->Lmx', lpq, c)
        lxx = numpy.einsum('Lmx,mx->Lx', lmx, c)
        lpx = lib.einsum('mp,Lmx->Lpx', co, lmx)
        jdiag += lpx[:, diag, diag].T.dot(lxx)
        kdiag += numpy.einsum('Lpx,Lpx->px', lpx, lpx)
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


def mrsf_hop(mf, hyb, spc, focka, fockb, jmo, kmo, cidx, oidx, vidx, occ_diag=None,
             singlet=True):
    '''MRSF part: forge collinear SF-TDA projected onto [G, D, L-R] + CV + OV + CO
    (singlet) or [L+R] + CV + OV + CO (triplet).'''
    o1, o2 = oidx
    nc, nv = len(cidx), len(vidx)
    nvirb = nv + 2
    vind_sf = _sf_vind(mf)
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
        occ_diag = _occ_diag(mf, nc + 2)
    # the Coulomb term (pp|qq) is large in spin-flip; without it Davidson can miss roots
    hd[noo:] = (fockb[cols[:, 1], cols[:, 1]] - focka[cols[:, 0], cols[:, 0]]
                - hyb * occ_diag[0][cols[:, 0], cols[:, 1]])
    if spc:
        hd[noo:] += cmat.diagonal()

    def mrsf_part(xm):
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
    return mrsf_part, hd


def ext_hop(mf, hyb, xc_cv, a00, focka, fockb, jmo, kmo, cidx, oidx, vidx, occ_diag=None,
            singlet=True):
    '''EMRSF extension: CV block (pyscf TDA on the closed-shell proxy) and the
    D coupling to the MRSF space, from batched AO J/K.'''
    o1, o2 = oidx
    nc, nv = len(cidx), len(vidx)
    ncv = nc * nv
    noo = 3 if singlet else 1
    nm = (nc + 2) * (nv + 2) - 4 + noo
    c = mf.mo_coeff
    cc_, cv_ = c[:, cidx], c[:, vidx]
    co1, co2 = c[:, [o1]], c[:, [o2]]

    mf_cv, f_cv, hyb_cv = cv_proxy(mf, xc_cv)
    mf_cv.mo_energy = f_cv.diagonal().copy()
    td_cv = tdscf.rhf.TDA(mf_cv)
    td_cv.singlet = singlet
    td_cv.verbose = 0
    vind_cv, hdiag_cv = td_cv.gen_vind()
    dsh = jmo[0, 0].diagonal() - jmo[1, 1].diagonal()
    sh_ia = (dsh[vidx][None, :] - dsh[cidx][:, None]).ravel()
    diag_cv = a00 - (1 - hyb_cv) * sh_ia
    # off-diagonal F' = F_cv + (1-hyb)[(pq|O2O2) - (pq|O1O1)] (cf. rohf_mrsf.cv_offdiag_fock)
    fcv = f_cv + (1 - hyb_cv) * (jmo[1, 1] - jmo[0, 0])
    fo_off = fcv[numpy.ix_(cidx, cidx)].copy()
    fv_off = fcv[numpy.ix_(vidx, vidx)].copy()
    fo_off[numpy.diag_indices_from(fo_off)] = 0
    fv_off[numpy.diag_indices_from(fv_off)] = 0
    if occ_diag is None:
        occ_diag = _occ_diag(mf, nc + 2)
    jdiag, kdiag = occ_diag[0][numpy.ix_(cidx, vidx)], occ_diag[1][numpy.ix_(cidx, vidx)]
    # A_ia,ia = e_ia + 2(ia|ia) - hyb (ii|aa) for singlets, e_ia - hyb (ii|aa) for triplets
    hd_cv = (hdiag_cv.reshape(nc + 1, nv + 1)[:-1, 1:].ravel() + diag_cv
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

    def d_parts(xm, xe):
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

    def cv_part(xe):
        nvec = xe.shape[0]
        z = numpy.zeros((nvec, nc + 1, nv + 1))
        z[:, :-1, 1:] = xe.reshape(nvec, nc, nv)
        y = numpy.asarray(vind_cv(z.reshape(nvec, -1))).reshape(nvec, nc + 1, nv + 1)
        y = y[:, :-1, 1:].reshape(nvec, -1) + xe * diag_cv
        x = xe.reshape(nvec, nc, nv)
        y += (lib.einsum('nia,ab->nib', x, fv_off) - lib.einsum('ij,nja->nia', fo_off, x)).reshape(nvec, -1)
        return y

    return d_parts, cv_part, hd_cv


def gen_tda_operation(mf, hyb, expansion=True, spc=True, xc_cv=None, singlet=True):
    '''Return (vind, hdiag) for the MRSF (expansion=False) or EMRSF matrix, singlet or
    triplet.'''
    _check_supported(mf, hyb)
    mf = ordered_scf(mf)[0]
    cidx, oidx, vidx = orb_indices(mf)
    focka, fockb = mo_fock(mf)
    jmo, kmo = _open_jk(mf, oidx)
    occ_diag = _occ_diag(mf, len(cidx) + 2)
    mrsf_part, hd = mrsf_hop(mf, hyb, spc, focka, fockb, jmo, kmo, cidx, oidx, vidx, occ_diag,
                             singlet)
    nm = hd.size
    if not expansion:
        def vind(xs):
            return mrsf_part(numpy.asarray(xs).reshape(-1, nm))
        return vind, hd

    o1, o2 = oidx
    a00 = fockb[o1, o1] - focka[o2, o2] - hyb * jmo[1, 1][o1, o1]
    d_parts, cv_part, hd_cv = ext_hop(mf, hyb, xc_cv, a00, focka, fockb, jmo, kmo,
                                      cidx, oidx, vidx, occ_diag, singlet)

    def vind(xs):
        xs = numpy.asarray(xs).reshape(-1, nm + hd_cv.size)
        xm, xe = xs[:, :nm], xs[:, nm:]
        dym, dye = d_parts(xm, xe)
        return numpy.hstack((mrsf_part(xm) + dym, cv_part(xe) + dye))
    return vind, numpy.concatenate((hd, hd_cv))


gen_hop = gen_tda_operation
