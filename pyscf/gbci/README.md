# GBCI, GBPDFT, and XMS-GBPDFT

This directory contains GBCI, GBPDFT, and multi-state (XMS) GBPDFT tools.

GBCI is the renamed and reorganized form of the previous SFNOCI module.  The
new name reflects that the implementation is no longer limited to the old
spin-flip NOCI interface and now serves as the grouped-bath CI reference used by
GBPDFT and XMS-GBPDFT.

## Dependencies

GBPDFT uses the compiled GBCI helper library for fast on-top pair-density
contractions. Build the PySCF-forge C extensions before running GBPDFT.

## Modules

Main GBCI modules:

- `pyscf.gbci.gbci`
- `pyscf.gbci.direct_gbci`
- `pyscf.gbci.rdm`
- `pyscf.gbci.fasscf`

GBPDFT modules:

- `pyscf.gbci.gbpdft`
- `pyscf.gbci.otpd`
- `pyscf.gbci.msgbpdft`
- `pyscf.gbci.xmsgbpdft`

## Import Changes

Use GBCI and GBPDFT through `pyscf.gbci`:

```python
from pyscf import gbci
from pyscf.gbci import gbpdft
from pyscf.gbci import otpd
```

Do not use the old SFNOCI or separate-package imports:

```python
from pyscf import sfnoci
from pyscf.sfnoci import sfnoci
```

## GBCI Usage

```python
from pyscf import gto, scf, gbci

mol = gto.M(
    atom="Li 0 0 0; H 0 0 1.6",
    basis="sto-3g",
    spin=0,
    verbose=0,
)

mf = scf.ROHF(mol).run()

mc = gbci.gbci(mf, ncas=2, nelecas=(1, 1))
mc.fcisolver.nroots = 1

e_gbci, e_cas, ci = mc.kernel()

print("GBCI energy:", e_gbci)
```

`group_a` can be used to build grouped baths by active MO index, atom index, or
occupation-pattern index:

```python
gbci.gbci(mf, 2, (1, 1), group_a={"mo": [[0], [1]]})
gbci.gbci(mf, 2, (1, 1), group_a={"atom": [[0], [1]]})
gbci.gbci(mf, 2, (1, 1), group_a={"occ": [[0], [1], [2]]})
```

## Solvers

Bath orbitals (FASSCF).  `mixed_routine` first runs the EDIIS + DIIS
solver (`FASSCF.ediis_kernel`, Kudin, Scuseria and Cances, J. Chem. Phys.
116, 8255 (2002)).  `optimize_mo` solves all baths of a GBCI run with it in
lockstep (`ediis_kernel_many`: one batched `get_jk` call per step).  A
converged bath is checked for a negative orbital-Hessian eigenvalue; at a
saddle point a step along that mode and a new EDIIS stage follow.  Baths that
EDIIS does not converge go through the damped iterations and Newton SOSCF as
before.  With `conv_tol_grad = None` EDIIS converges the bath gradient to
`1e-3 sqrt(conv_tol)`, not below 1e-8 (DIIS converges fast near the
solution); quantities that are linear in the bath orbitals (XMS energies,
nuclear gradients) then do not depend on the solver path.  Options:

```python
mc._fasscf.mixed_routine_options = {"ediis": False}   # previous solver only
mc._fasscf.ediis_switch = 0.1         # EDIIS -> DIIS on max |FP - PF|
mc._fasscf.ediis_occupation = "aufbau"  # or "mom", "imom"
mc._fasscf.max_stability = 5          # 0: no stability check
mc._fasscf.stability_tol = 1e-4
```

CI solver (`GBCISolver`).  H|CI> is computed by the libgbci kernel over the
signed excitation lists (`sigma_kernel = "contract_h"`); `"block"` computes it
by group blocks with the PySCF FCI kernels, which is slower for large active
spaces.  The Davidson solver uses the exact Hamiltonian on
a pspace block of `pspace_size` determinants (completed to whole
spin-coupling sets, with the `fix_spin_` penalty) as preconditioner and guess
(generalized Davidson); spaces of at most `pspace_size` determinants are
diagonalized directly unless `davidson_only`.  `pspace_size = 0` gives the
diagonal preconditioner.  `guess_noise` (1e-3) adds a random part of that
norm to the guess vectors.

## GBPDFT Usage

Import the GBPDFT module from `pyscf.gbci`:

```python
from pyscf.gbci import gbpdft
```

### Build GBPDFT directly from an SCF object

```python
from pyscf import gto, scf
from pyscf.gbci import gbpdft

mol = gto.M(
    atom="Li 0 0 0; H 0 0 1.6",
    basis="sto-3g",
    spin=0,
    verbose=0,
)

mf = scf.ROHF(mol).run()

pdft = gbpdft.GBCI(
    mf,
    "tPBE",
    ncas=2,
    nelecas=(1, 1),
)

pdft.fcisolver.nroots = 1

e_tot, e_ot, e_gbci, e_cas, ci, mo_coeff, mo_energy = pdft.kernel()

print("GBPDFT total energy:", e_tot)
print("on-top energy:", e_ot)
print("GBCI energy:", e_gbci)
```

### Build GBPDFT from an existing GBCI object

```python
from pyscf import gto, scf, gbci
from pyscf.gbci import gbpdft

mol = gto.M(
    atom="Li 0 0 0; H 0 0 1.6",
    basis="sto-3g",
    spin=0,
    verbose=0,
)

mf = scf.ROHF(mol).run()

mc = gbci.gbci(mf, ncas=2, nelecas=(1, 1))
mc.fcisolver.nroots = 1

pdft = gbpdft.gbci(mc, "tPBE")

e_tot, e_ot, e_gbci, e_cas, ci, mo_coeff, mo_energy = pdft.kernel()

print("GBPDFT total energy:", e_tot)
```

## XMS-GBPDFT Usage

XMS-GBPDFT is built from a GBPDFT object with `multi_state`:

```python
import numpy as np
from pyscf import gto, scf
from pyscf.gbci import gbpdft

mol = gto.M(
    atom="Li 0 0 0; H 0 0 1.6",
    basis="sto-3g",
    spin=0,
    verbose=0,
)

mf = scf.ROHF(mol).run()

pdft = gbpdft.GBCI(
    mf,
    "tPBE",
    ncas=2,
    nelecas=(1, 1),
    group_a={"mo": [[0], [1]]},
)
pdft.fcisolver.nroots = 2

xms = pdft.multi_state(np.ones(2) / 2, "xms")
e_tot, e_ot, e_gbci, e_cas, ci, mo_coeff, mo_energy = xms.kernel()

print("XMS-GBPDFT average energy:", e_tot)
print("XMS-GBPDFT state energies:", xms.e_states)
print("XMS-GBPDFT GBCI reference energies:", e_gbci)
print("XMS-GBPDFT effective Hamiltonian:", xms.get_heff_pdft())
```

## Examples

Example input files are available in:

```text
examples/gbci/
```

For example:

```bash
python examples/gbci/00-gbci.py
python examples/gbci/01-gbpdft.py
python examples/gbci/02-xmsgbpdft.py
```
