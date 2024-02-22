# sdstate
Implementation of Slater Determinant states as dictionaries of states and coefficient pairs. Each state is represented with an integar treated as binary, allowing applying of Excitation operators on the state efficiently.
# fstate
Similar implementation as sdstate, Slater Determinant represented as binary str, abondoned later due to inefficiency in operation.
# Lanczos
Memory-efficient implementation of Lanczos iteration for estimating Hamiltonian spectrum range.

Data structure: sdstate and fstate, dictionary implementation of slater determinant states, with integar or binary string as hash key

Accepts Hamiltonian as openfermion.FermionOperator, compute the energy expectation by applying the Hamiltonian to the state

Demo as in lanczos_demo.ipynb


