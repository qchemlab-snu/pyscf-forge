#!/usr/bin/env python

'''
MRSF-TDA for H2O with BHHLYP / cc-pVDZ.

The reference is the triplet (M_S = 1) ROKS state. Singlet MRSF includes the ground
state S0, so excitation energies are taken relative to the lowest singlet.
'''

from pyscf import gto, dft, mrsftda
from pyscf.data.nist import HARTREE2EV

mol = gto.M(atom='''
O  0.000000000  0.000000000 -0.041061554
H -0.533194329  0.533194329 -0.614469223
H  0.533194329 -0.533194329 -0.614469223''',
            basis='cc-pvdz', spin=2)

# BHHLYP as in GAMESS and OpenQP: 50% HF + 50% B88 exchange, LYP correlation
mf = dft.ROKS(mol, xc='HF*0.5+0.5*B88, 1.0*LYP')
mf.kernel()

# Singlets
td = mrsftda.TDA_MRSF(mf)          # or mf.TDA_MRSF()
td.kernel(nstates=5)
td.analyze()
f = td.oscillator_strength()

# Triplets
td_t = mrsftda.TDA_MRSF(mf)
td_t.singlet = False
td_t.kernel(nstates=5)

print(f'\nMRSF, H2O BHHLYP/cc-pVDZ, reference (triplet ROKS) {td.e_ref:.8f} Eh')
print('       E (Eh)          dE vs S0 (eV)   f')
for i, et in enumerate(td.e_tot):
    fi = f'{f[i - 1]:.4f}' if i > 0 else ''
    print(f'  S{i}  {et:14.8f}  {(td.e[i] - td.e[0]) * HARTREE2EV:10.4f}  {fi:>8s}')
for i, et in enumerate(td_t.e_tot):
    print(f'  T{i + 1}  {et:14.8f}  {(td_t.e[i] - td.e[0]) * HARTREE2EV:10.4f}')
