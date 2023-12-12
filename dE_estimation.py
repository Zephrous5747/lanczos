import pandas as pd
from ast import literal_eval
import h5py
import numpy as np
from openfermion import InteractionOperator
from openfermion.transforms import get_fermion_operator
from sdstate import *

if __name__ == "__main__":
#     Steps to run in Lanczos iteration. Recommand to use no more than 20 for large systems of ~100 qubits
    steps = 10
#     Running multiprocessing of Hartree-Fock estimation. Running N threads in parallel for an n-qubit system.
    parallelization = True
    
    instance_info_cat = []
    path = "hamiltonians_catalysts/"
    file_name = "7_melact_6-311++G___18_116c0129-525c-456b-a6ac-34b830f64bc3.hdf5"
    with h5py.File(path + file_name, mode="r") as h5f:
        attributes = dict(h5f.attrs.items())
        one_body = np.array(h5f["one_body_tensor"])
        two_body = np.array(h5f["two_body_tensor"])
        hamiltonian = InteractionOperator(constant=attributes['constant'], one_body_tensor=one_body, two_body_tensor=two_body)
        Hf = get_fermion_operator(hamiltonian)
    print("n_qubit = {}".format(one_body.shape[0]))
    max_state, min_state, HF_max, HF_min = HF_spectrum_range(Hf, multiprocessing = parallelization)
    HF_dE = HF_max - HF_min
    print("Hartree-Fock dE = {}".format(HF_dE))
    print("Hartree-Fock maximum Slater Determinant state: {}".format(max_state))
    print("Hartree-Fock minimum Slater Determinant state: {}".format(min_state))
    lanczos_max, lanczos_min = lanczos_total_range(Hf, steps = steps, states = [max_state, min_state])
    lanczos_dE = lanczos_max-lanczos_min
    print(lanczos_max,lanczos_min)
    print("Lanczos algorithm with {} iterations, dE = {}".format(steps, lanczos_dE))