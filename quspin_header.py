import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import X, Y, Z, RX, RY, RZ, CNOT, H, to_matrix_gate, merge, add
from qulacs import Observable
#from qulacsvis import circuit_drawer
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import obs_vs_time

def add_RZZ_gate(circuit,sub_qubits,param):
    '''
    add exp(i\alpha ZZ/2) to the circuit
    input:
    - circuit: quantum circuit as QuantumCircuit
    - sub_qubits: indices of qubits on which RZZ gate acts
    - param: angle (\alpha)
    '''
    circuit.add_CNOT_gate(sub_qubits[0],sub_qubits[1])
    circuit.add_RZ_gate(sub_qubits[1],param)
    circuit.add_CNOT_gate(sub_qubits[0],sub_qubits[1])


def make_ansatz_HVA(params,num_qubits,num_layers):
    '''
    construct Hamiltonian variational ansatz
    - input
        - num_qubits: number of qubits (N)
        - num_layers: number of layers (L)
        - params: list parameters (number of parameters: (3*num_qubits-2)*num_layers)
    - output: ansatz as QuantumCircuit
    '''
    circuit = QuantumCircuit(num_qubits)
    #initialization to be in correct mag sector
    # for qubit in range(num_qubits):
    #     circuit.add_H_gate(qubit)
    params_index = 0
    for layer in range(num_layers):
        # #RXY even
        # for qubit in range(0,num_qubits-1,2):
        #     add_RXY_gate(circuit,[qubit,qubit+1],params[params_index])
        #     params_index += 1
        # #RXY odd
        # for qubit in range(1,num_qubits-1,2):
        #     add_RXY_gate(circuit,[qubit,qubit+1],params[params_index])
        #     params_index += 1
        
        #RZZ even
        for qubit in range(0,num_qubits-1,2):
            add_RZZ_gate(circuit,[qubit,qubit+1],params[params_index])
            params_index += 1
        #RZZ odd
        for qubit in range(1,num_qubits-1,2):
            add_RZZ_gate(circuit,[qubit,qubit+1],params[params_index])
            params_index += 1
        #RX
        for qubit in range(num_qubits):
            circuit.add_RX_gate(qubit,params[params_index])
            params_index += 1
    return circuit
    
def compute_obs_ED(psi_init,num_qubits,j_coupling,h_coupling,time_max,num_steps=1000):
    ''' 
    compute observables via ED
    input:
        - psi_init: initial state
        - num_qubits: number of qubits
        - j_coulpling: ZZ coefficient
        - h_coupling: X coefficient
        - time_max: final time
        - num_steps: number of steps
    output:
        - psi_t: time evolved state [psi(0),psi(dt), psi(2*dt),...]
        - mag_list_ED: list of magnetization
    '''

    basis = spin_basis_1d(num_qubits) #state basis 

    #set terms in Hamiltonian 
    h_zz = [[-j_coupling , i ,i+1] for i in range(num_qubits-1)]
    h_x = [[-h_coupling, i] for i in range(num_qubits)]

    # static and dynamic lists
    static = [["zz",h_zz],["x",h_x]] 
    dynamic=[]
    # # Hfull = hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
    H = hamiltonian(static,dynamic,basis=basis,dtype=np.float64,check_herm=False,check_pcon=False,check_symm=False)

    #construct magnetization operator
    mag_Z = [[1/num_qubits, i] for i in range(num_qubits)]
    static = [['z',mag_Z]]
    dynamic = []
    mag_op = hamiltonian(static,dynamic,basis=basis,dtype=np.float64,check_herm=False,check_pcon=False,check_symm=False)

    #list of times
    times = np.linspace(0.0,time_max,num_steps)

    psi_t = H.evolve(psi_init,0.0,times)
    # obs_time = obs_vs_time(psi_t,times,dict(mag=mag_op,efield=efield_op,cc=cc_op))
    obs_time = obs_vs_time(psi_t,times,dict(mag=mag_op))
    mag_list_ED = obs_time['mag']

    return psi_t, mag_list_ED #, efield_list_ED, cc_list_ED

def compute_state(params,num_qubits,num_layers):
    """ 
    compute state for a given set of params
    input:
        - params: parameters
        - num_qubits: number of qubits
        - num_layers: number of layers
    output: statevector
    """
    #make ansatz
    ansatz = make_ansatz_HVA(params,num_qubits,num_layers)

    #make initial satate |0>
    state = QuantumState(num_qubits)
    state.set_zero_state()

    #compute statevector
    ansatz.update_quantum_state(state)
    state_vector = state.get_vector()
    return state_vector

def make_mapping_reverse(num_qubits):
    ''' 
    make a list of inversed binary
    ex: for 3q case,
    [000,001,010,011,...,111] -> [000,100,010,110,...,111]
    '''
    num_elements = 2**num_qubits #number of vector elements
    index_list = range(num_elements) #list of index
    index_rev_list = [] 
    for index in index_list:
        index_bin = bin(index)[2:] #translate index into binary
        index_bin = index_bin.zfill(num_qubits) #zerofill to align the length of binary
        index_bin_rev = ''.join(list(reversed(index_bin))) #reversed binary
        index_rev =  int(index_bin_rev,2) #translate reversed binary into decimal
        index_rev_list.append(index_rev) 
    return index_rev_list
    
def map_state_reverse(state):
    ''' 
    reorder states in different convention
    ex: for 3q case,
    [C_{000},C_{001},C_{010},C_{011},...,C_{111}] -> [C_{000},C_{100},C_{010},C_{110},...,C_{111}]
    '''
    num_elements = len(state)
    num_qubits = int(np.log2(num_elements))
    index_rev_list = make_mapping_reverse(num_qubits)
    state_rev = np.zeros(num_elements,dtype = 'complex_')
    for index,index_rev in enumerate(index_rev_list):
        state_rev[index] = state[index_rev]
    return state_rev
    
def compute_state(params,num_qubits,num_layers):
    """ 
    compute state for a given set of params
    input:
        - params: parameters
        - num_qubits: number of qubits
        - num_layers: number of layers
    output: statevector
    """
    #make ansatz
    ansatz = make_ansatz_HVA(params,num_qubits,num_layers)

    #make initial satate |0>
    state = QuantumState(num_qubits)
    state.set_zero_state()

    #compute statevector
    ansatz.update_quantum_state(state)
    state_vector = state.get_vector()
    return state_vector

def getPMag_list(params_init,num_layers,num_qubits,j_coupling,h_coupling,time_max,num_steps):
    state_init = compute_state(params_init,num_qubits,num_layers)
    state_init_rev = map_state_reverse(state_init)
    psi_t_ED, mag_list_ED = compute_obs_ED(state_init,num_qubits,j_coupling,h_coupling,time_max,num_steps=num_steps)
    
    return psi_t_ED, mag_list_ED