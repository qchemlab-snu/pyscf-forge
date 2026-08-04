#!/usr/bin/env python
#
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0

import unittest

from pyscf import gbci, gto, lib, scf


lib.num_threads(1)

BASIS = "cc-pvdz"
BOND_LENGTH = 1.5
GROUP_A = {"atom": [0]}

LIH_GRAD_Z = 0.0183010294
LIF_GRAD_Z = 0.0354853261


def close_scf_resources(mf):
    chkfile = getattr(mf, "_chkfile", None)
    if chkfile is not None:
        chkfile.close()
        mf._chkfile = None
    if hasattr(mf, "chkfile"):
        mf.chkfile = None


def close_gbci_resources(mc):
    close_scf_resources(mc)
    fasscf = getattr(mc, "_fasscf", None)
    if fasscf is not None:
        close_scf_resources(fasscf)


def get_gbci_grad(atom, ncas, nelecas):
    mol = gto.M(
        atom=atom,
        basis=BASIS,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mc = None
    try:
        mf.conv_tol = 1e-12
        mf.run()

        mc = gbci.gbci(mf, ncas, nelecas, group_a=GROUP_A)
        mc.fcisolver.conv_tol = 1e-10
        mc.run()
        return mc.nuc_grad_method().kernel()
    finally:
        if mc is not None:
            close_gbci_resources(mc)
        close_scf_resources(mf)


class KnownValues(unittest.TestCase):
    def test_lih_2o2e(self):
        grad = get_gbci_grad(
            f"Li 0 0 0; H 0 0 {BOND_LENGTH}", 2, (1, 1))
        self.assertAlmostEqual(float(grad[0, 2]), LIH_GRAD_Z, 9)

    def test_lif_4o4e(self):
        grad = get_gbci_grad(
            f"Li 0 0 0; F 0 0 {BOND_LENGTH}", 4, (2, 2))
        self.assertAlmostEqual(float(grad[0, 2]), LIF_GRAD_Z, 9)


if __name__ == "__main__":
    print("Full Tests for GBCI nuclear gradients")
    unittest.main()
