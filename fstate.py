"""Defines an implementation of the Slater Determinant states,with a dictionary to represent the occupied states with the corresponding constants.
"""
import numpy as np
from itertools import product
import copy
from scipy.linalg import eigh_tridiagonal
from multiprocessing import Pool
import openfermion as of

class fstate:
    dic = {}
    n_qubit = 0
    def __init__(self, s = None, coeff = 1):
        self.dic = {}
        if s:
            self.dic[s] = coeff
            self.n_qubit = len(s)
            
    def norm(self):
#         Return the norm of the current state
        return np.sqrt(self @ self)

    def normalize(self):
#         Normalize the current state
        n = self.norm()
        for i in self.dic:
            self.dic[i] /= n
        return None
    
    def __add__(self, other):
        if self.n_qubit != 0:
            assert self.n_qubit == other.n_qubit, "qubit number mismatch"
        else:
            self.n_qubit = other.n_qubit
        result = copy.deepcopy(self)
        for s in other.dic:
            if s in result.dic:
                result.dic[s] += other.dic[s]
            else:
                result.dic[s] = other.dic[s]
        return result
    
    def __sub__(self, other):
        return self + (-1) * other
    
    def __mul__(self, n):
#         Defines constant multiplication
        result = copy.deepcopy(self)
        for s in result.dic:
            result.dic[s] *= n
        return result
    
    def __rmul__(self, n):
        return self.__mul__(n)
    
    def __truediv__(self, n: int):
        return self.__mul__(1/n)
    
    def __matmul__(self, other):
        if isinstance(other, of.FermionOperator):
            return self.fermop_state(other)
        if isinstance(other, np.ndarray):
            return self.matrix_state(other)
        if self.n_qubit == 0 or other.n_qubit == 0:
            return 0
        count = 0
        assert self.n_qubit == other.n_qubit, "qubit number mismatch"
        lis = list(set(list(self.dic.keys())) & set(list(other.dic.keys())))
        for s in lis:
            count += np.conjugate(self.dic[s]) * other.dic[s]
        return count 
    
    def __str__(self):
        return str(self.dic)
    
    def exp(self, Htbt: np.ndarray):
#         Return the expectation value Hamiltonian on the current state with Hamiltonian in the two-body-tensor form
        return np.real(self @ Htbt @ self)
        
    
    def matrix_state(self, Htbt: np.ndarray):
        """
        return matrix state product of state with matrix H, for H represented as the two-body tensor, and 
        return H|s>
        """
        n = Htbt.shape[0]
        assert self.n_qubit == n or self.n_qubit == 0, "Qubit number {} mismatch size of Htbt {}".format(self.n_qubit, n) 
        re_state = fstate()
        for s in self.dic:
            for i, j, k, l in product(range(n), repeat=4):
                if Htbt[i,j,k,l] != 0:
                    cur_state = [int(i) for i in s]
                    coe = 1
                    if cur_state[l] != 1:
                        continue
                    cur_state[l] -= 1
                    coe *= (-1) ** sum(cur_state[:l])
                    if cur_state[k] != 0:
                        continue
                    cur_state[k] += 1
                    coe *= (-1) ** sum(cur_state[:k])
                    if cur_state[j] != 1:
                        continue
                    cur_state[j] -= 1
                    coe *= (-1) ** sum(cur_state[:j])
                    if cur_state[i] != 0:
                        continue
                    cur_state[i] += 1
                    coe *= (-1) ** sum(cur_state[:i])
                    tmp = ''.join([str(i) for i in cur_state])
                    re_state += fstate(tmp, coe * self.dic[s] * Htbt[i,j,k,l])
        return re_state 
    
    
def HF_energy(Htbt, n, ne):
    """Find the energy of largest and smallest slater determinant states with Htbt as Hamiltonian 2e tensor and 
     number of electrons as ne.
    """    
    low_str = ne * "1" + (n - ne) * "0"
    lstate = fstate(low_str)
    E_low = lstate.exp(Htbt)
    high_str = (n - ne) * "0" + ne * "1"
    hstate = fstate(high_str)
    E_high = hstate.exp(Htbt)
    return E_high, E_low

def HF_spectrum_range(Htbt, multiprocessing = True):
    """Compute the Hatree-Fock energy range of the Hamiltonian 2e tensor Htbt for all number of electrons.
    Multiprocessing parameter is set to parallelize computations for the states with different number of electrons
    """
    n = Htbt.shape[0]
    if multiprocessing:
        with Pool() as pool:
            res = pool.starmap(HF_energy, [(Htbt, n, ne) for ne in range(n)])
        low = 1e10
        low_state = ""
        high = -1e10
        high_state = ""
        for ne in range(len(res)):
            E_high = res[ne][0]
            E_low = res[ne][1]
            if E_low < low:
                low_state = ne * "1" + (n - ne) * "0"
                low = E_low
            if E_high > high:
                high_state = (n - ne) * "0" + ne * "1"
                high = E_high
    else:
        low = 1e10
        low_state = ""
        high = -1e10
        high_state = ""
        for ne in range(n):
            low_str = ne * "1" + (n - ne) * "0"
            lstate = fstate(low_str)
    #         <low|H|low>
            E_low = lstate.exp(Htbt)
    #         print(E_low)
            high_str = (n - ne) * "0" + ne * "1"
            hstate = fstate(high_str)
    #         <high|H|high>
            E_high = hstate.exp(Htbt)
            if E_low < low:
                low_state = low_str
                low = E_low
            if E_high > high:
                high_state = high_str
                high = E_high
#             low = min(np.real(E_low), low)
#             high = max(np.real(E_high), high)
#         print("Highest state:{}".format(high_state))
#         print("Lowest state:{}".format(low_state))
    print("HF E_max: {}".format(high))
    print("HF E_min: {}".format(low))
    return high_state, low_state, high, low


def lanczos(Htbt: np.ndarray, steps, state = None, ne = None):
    """Applies lanczos iteration on the given 2e tensor Htbt, with number of steps given by steps,
    with initial state as input or number of electrons as input ne.
    Returns normalized states in each iteration, and a tridiagonal matrix with main diagonal in A and sub-diagonal in B.
    """
    n_qubits = Htbt.shape[0]
    if state == None:
        if ne == None:
            ne = n_qubits // 2
        state = fstate("1"*ne + "0"*(n_qubits - ne))
        
    tmp = state @ Htbt
    ai = tmp @ state
    tmp -= ai * state
    A = [ai]
    B = []
    states = [state]
    vi = tmp
    for i in range(1,steps):
        bi = tmp.norm()
        if bi != 0:
            vi = tmp / bi
        tmp = vi @ Htbt
        ai = vi @ tmp
        tmp -= ai * vi 
        tmp -= bi * states[i - 1]
        states.append(vi)
        A.append(ai)
        B.append(bi)
    return states, A, B

def lanczos_range(Htbt, steps, state = None, ne = None):
    """ Returns the largest and the smallest eigenvalue from Lanczos iterations with given number of steps,
    number of electrons or initial state.
    """
    _, A, B = lanczos(Htbt, steps = steps, state = state, ne = ne)
    eigs, _ = eigh_tridiagonal(A,B)
    return max(eigs), min(eigs)

def lanczos_total_range(Htbt, steps, states = [], multiprocessing = True):
    """ Returns the largest and the smallest eigenvalue from Lanczos iterations with given number of steps,
    for all possible number of electrons. Multiprocessing will parallelize the computation for all possible 
    number of electrons. State is in the form of list of binary strings, indicating the maximum and minimum 
    HF energy states to start the iteration
    """
    if states != []:
        [max_str, min_str] = states
        max_state = fstate(max_str)
        min_state = fstate(min_str)
        E_max, _ = lanczos_range(Htbt, steps = steps, state = max_state)
        _, E_min = lanczos_range(Htbt, steps = steps, state = min_state)
    else:
        if multiprocessing:
            with Pool() as pool:
                res = pool.starmap(lanczos_range, [(Htbt, steps, None, ne) for ne in range(Htbt.shape[0])])
            E_max = max([i[0] for i in res])
            E_min = min(i[1] for i in res)
        else:
            E_max = -1e10
            E_min = 1e10
            for ne in range(Htbt.shape[0]):
                states, A, B = lanczos(Htbt, steps = steps, ne = ne)
                eigs, _ = eigh_tridiagonal(A,B)
                E_max = max(max(eigs), E_max)
                E_min = min(min(eigs), E_min)
    return E_max, E_min

def MP2_correction(atbt, obt_eigs, state):
    """Defines an implementation for computing the MP2 correction, from onebody operator obt and two-body operator 
    atbt, and the given state in a binary string.
    """

    n = len(state)
    occupied = [i for i in range(n) if state[i] == "1"]
    virtual = [i for i in range(n) if state[i] == "0"]
    tmp = 0
    for i,j in product(occupied, repeat = 2):
        for a, b in product(virtual, repeat = 2):
            k = obt_eigs[i] + obt_eigs[j] - obt_eigs[a] - obt_eigs[b]
            if abs(k) > 1e-14:
                tmp += atbt[i,j,a,b] ** 2 / k
    return tmp/4