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
Mixed-reference spin-flip (MRSF) and extended MRSF (EMRSF) TDA.

    from pyscf import mrsf
    td = mrsf.TDA_EMRSF(mf)     # mf: converged triplet ROHF/ROKS
    td.kernel()
'''

from pyscf.mrsf import rohf_mrsf
from pyscf.mrsf.rohf_mrsf import get_ab_mrsf, get_ab


def TDA_MRSF(mf):
    return mf.remove_soscf().TDA_MRSF()


def TDA_EMRSF(mf):
    return mf.remove_soscf().TDA_EMRSF()
