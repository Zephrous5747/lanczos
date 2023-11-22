import sys
sys.path.append("../")
import pickle
import io
from fstate import *

import saveload_utils as sl
import ferm_utils as feru
import csa_utils as csau
import var_utils as varu
import openfermion as of
import numpy as np
import scipy
from openfermion import FermionOperator
import copy
import time

if __name__ == "__main__":
    lis = []
    
    default_mols=  ["h2", "h4", "lih", "h2o", "nh3", "n2"]
    mols = default_mols if len(sys.argv) < 2 else [sys.argv[1]]
    for mol in mols:
        print(mol)
        # Get two-body tensor
        Hf = sl.load_fermionic_hamiltonian(mol)
        
        spin_orb = of.count_qubits(Hf)
        spatial_orb = spin_orb // 2
        spin_flag = True
        Htbt = feru.get_chemist_tbt(Hf, spin_orb, spin_orb = spin_flag)
        one_body = varu.get_one_body_correction_from_tbt(Hf, feru.get_chemist_tbt(Hf))
        onebody_matrix = feru.get_obt(one_body, n = spin_orb, spin_orb = spin_flag)
        Htbt += feru.onebody_to_twobody(onebody_matrix)
        
        E_max, E_min = lanczos_total_range(Htbt, steps = 2, multiprocessing = True)
        print("Lanczos E_max: {}".format(E_max))
        print("Lanczos E_min: {}".format(E_min))
        dE = E_max - E_min
        print("Lanczos dE: {}".format(dE))
        
        
        HF_max, HF_min, HF_dE = HF_spectrum_range(Htbt)
        print("HF dE: {}".format(HF_dE))
        deltaE = dE - HF_dE
        pdeltaE = deltaE / dE
        l = [mol, round(dE, 2), round(HF_dE, 2), round(deltaE, 2), "{}%".format(round(pdeltaE, 2))]
        lis.append(l)
        print(l)
        if len(sys.argv) < 2:
            with open("./HF_results.pkl", "wb") as f:
                pickle.dump(lis, f)
        else:
            with open("./HF_result-{}.pkl".format(sys.argv[1]), "wb") as f:
                pickle.dump(lis, f)
        
        