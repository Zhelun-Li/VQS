import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from qiskit.primitives import Sampler,Estimator
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister,ClassicalRegister
from qiskit.quantum_info import SparsePauliOp


from qiskit.algorithms.gradients import ParamShiftEstimatorGradient
from qiskit.circuit import ParameterVector
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def add_RZZ_gate(circuit,sub_qubits,param):
    '''
    add exp(i\alpha ZZ/2) to the circuit
    input:
    - circuit: quantum circuit as QuantumCircuit
    - sub_qubits: indices of qubits on which RZZ gate acts
    - param: angle (\alpha)
    '''
    circuit.cx(sub_qubits[0],sub_qubits[1])
    circuit.rz(param,sub_qubits[1])
    circuit.cx(sub_qubits[0],sub_qubits[1])
    
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
    params_index = 0
    for layer in range(num_layers):

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
            circuit.rx(params[params_index],qubit)
            params_index += 1
    return circuit


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
    params_index = 0
    for layer in range(num_layers):

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
            circuit.rx(params[params_index],qubit)
            params_index += 1
    return circuit

def make_ansatz_HVA_weights(n_params,num_qubits,num_layers):
    '''
    construct Hamiltonian variational ansatz
    - input
        - num_qubits: number of qubits (N)
        - num_layers: number of layers (L)
        - params: list parameters (number of parameters: (3*num_qubits-2)*num_layers)
    - output: ansatz as QuantumCircuit
    '''
    circuit = QuantumCircuit(num_qubits)
    weights = ParameterVector("weight", n_params)
    params_index = 0
    for layer in range(num_layers):

        #RZZ even
        for qubit in range(0,num_qubits-1,2):
            add_RZZ_gate(circuit,[qubit,qubit+1],weights[params_index])
            params_index += 1
        #RZZ odd
        for qubit in range(1,num_qubits-1,2):
            add_RZZ_gate(circuit,[qubit,qubit+1],weights[params_index])
            params_index += 1
        #RX
        for qubit in range(num_qubits):
            circuit.rx(weights[params_index],qubit)
            params_index += 1
    return circuit


def add_conditional_RX_gate(circuit,qubit_ancilla, qubit_target, param, params_index,deriv_index_list):
    ''' 
    if params_index is equal to one element in deriv_index_list, then add RZ.CZ to circuit (instead of RZ)
    - input:
        - circuit: quantum circuit
        - qubit_ancilla: index of ancilla qubit
        - qubit_target: index of target qubit (on which RZ acts on)
        - param: rotation angle
        - params_index: index of parameter
        - deriv_index_list: list of index [i,j] wrt which we take derivative (i<j)
    -output: None
    '''
    #derivative wrt RZ gate
    if deriv_index_list[0] == params_index or deriv_index_list[1] == params_index:
        circuit.x(qubit_ancilla) #add X gate
        circuit.cx(qubit_ancilla,qubit_target) #add CX gate 
    circuit.rx(param,qubit_target) #add RX gate
    
def add_conditional_RZZ_gate(circuit,qubit_ancilla, qubits_target, param, params_index,deriv_index_list):
    ''' 
    if params_index is equal to one element in deriv_index_list, then add RZ.CZ to circuit (instead of RZ) two between CNOTs
    - input:
        - circuit: quantum circuit
        - qubit_ancilla: index of ancilla qubit
        - qubits_target: list of indices of target qubits (on which RZ acts on)
        - param: rotation angle
        - params_index: index of parameter
        - deriv_index_list: list of index [i,j] wrt which we take derivative (i<j)
    -output: None
    '''
    #derivative wrt RZ gate
    circuit.cx(qubits_target[0],qubits_target[1])
    if deriv_index_list[0] == params_index or deriv_index_list[1] == params_index:
        circuit.x(qubit_ancilla) #add X gate
        circuit.cz(qubit_ancilla,qubits_target[1]) #add CZ gate 
    circuit.rz(param,qubits_target[1]) #add RZ gate
    circuit.cx(qubits_target[0],qubits_target[1])
    
def make_M_circuit_HVA(params,num_qubits,num_layers,deriv_index_list):
    '''
    circuit for M_{ij}
    - input
        - num_qubits: number of qubits (N)
        - num_layers: number of layers (L)
        - params: list parameters (\theta) (number of parameters: 2*num_qubits*num_layers)
        - deriv_index_list (list): indices for parameters which we want to take derivative with respect to. ([i,j] for M_{ij}. (i<j))
    - output: quantum circuit
    '''
    circuit = QuantumCircuit(num_qubits+1,1) #make quantum circuit with N-registers and 1-ancilla, and a classical qubit
    qubit_ancilla = num_qubits #ancilla in last qubit

    #make initial state in physical qubits
    # for qubit in range(num_qubits):
    #     circuit.add_H_gate(qubit)

    circuit.h(qubit_ancilla) # make (|0>+|1>)/sqrt(2) in ancilla
    # circuit.add_S_gate(qubit_ancilla) # make (|0>+i|1>)/sqrt(2) as an initial state (eta=1 in Li-Benjamin) in ancilla

    params_index = 0
    for layer in range(num_layers):
        #RZZ even
        for qubit in range(0,num_qubits-1,2):
            add_conditional_RZZ_gate(circuit,qubit_ancilla, [qubit,qubit+1], params[params_index], params_index,deriv_index_list)
            # add_RZZ_gate(circuit,[qubit,qubit+1],params[params_index])
            params_index += 1
        #RZZ odd
        for qubit in range(1,num_qubits-1,2):
            add_conditional_RZZ_gate(circuit,qubit_ancilla, [qubit,qubit+1], params[params_index], params_index,deriv_index_list)
            # add_RZZ_gate(circuit,[qubit,qubit+1],params[params_index])
            params_index += 1
        #RX 
        for qubit in range(num_qubits):
            add_conditional_RX_gate(circuit,qubit_ancilla, qubit, params[params_index], params_index,deriv_index_list)
            # circuit.add_RZ_gate(qubit,params[params_index])
            params_index += 1

    circuit.h(qubit_ancilla) #basis rotation for measurements in {|+>, |->} basis 
    circuit.measure(qubit_ancilla,0)
    return circuit

def make_V_circuit_HVA(params,num_qubits,num_layers,deriv_index):
    '''
    circuit for V_{i}
    - input
        - num_qubits: number of qubits (N)
        - num_layers: number of layers (L)
        - params: list parameters (\theta) (number of parameters: 2*num_qubits*num_layers)
        - deriv_index: index of a parameter with respect to which we want to take derivative. (i for V_{i})
    - output: dictionary of quantum circuit
        - key: terms in Hamiltonian (str)
        - value: quantum circuit corresponding to the key
    '''
    circuit = QuantumCircuit(num_qubits+1,1) #N-register + ancilla
    qubit_ancilla = num_qubits #ancilla in last qubit

    #make initial state in physical qubits
    # for qubit in range(num_qubits):
    #     circuit.add_H_gate(qubit)
        
    circuit.h(qubit_ancilla) # make (|0>+|1>)/sqrt(2) in ancilla qubit #changed, Dec. 12, 2022

    deriv_index_list = [deriv_index,len(params)*10]
    params_index = 0
    for layer in range(num_layers):
        #RZZ even
        for qubit in range(0,num_qubits-1,2):
            add_conditional_RZZ_gate(circuit,qubit_ancilla, [qubit,qubit+1], params[params_index], params_index,deriv_index_list)
            # add_RZZ_gate(circuit,[qubit,qubit+1],params[params_index])
            params_index += 1
        #RZZ odd
        for qubit in range(1,num_qubits-1,2):
            add_conditional_RZZ_gate(circuit,qubit_ancilla, [qubit,qubit+1], params[params_index], params_index,deriv_index_list)
            # add_RZZ_gate(circuit,[qubit,qubit+1],params[params_index])
            params_index += 1
    
        #RZ 
        for qubit in range(num_qubits):
            add_conditional_RX_gate(circuit,qubit_ancilla, qubit, params[params_index], params_index,deriv_index_list)
            # circuit.add_RZ_gate(qubit,params[params_index])
            params_index += 1

    #make quantum circuits for each term {h_j} in Hamiltonian
    #circuit_dict = {}
    circuit_dict=[]
    #X term in Hamiltonian
    for qubit in range(num_qubits):
        #term = f"X {qubit}"
        circuit_x = circuit.copy() #copy circuit
        circuit_x.x(qubit_ancilla) #add X gate
        circuit_x.cx(qubit_ancilla,qubit) #add control-h_j (here h_j = X)
        circuit_x.h(qubit_ancilla) #basis rotation for measurements in {|+>, |->} basis
        circuit_x.measure(qubit_ancilla,0)
        circuit_dict.append(circuit_x)

    #ZZ term in Hamiltonian
    for qubit in range(num_qubits-1):
        #term = f"Z {qubit} Z {qubit+1}"
        circuit_zz = circuit.copy() #copy circuit
        circuit_zz.x(qubit_ancilla) #add X gate
        circuit_zz.cz(qubit_ancilla,qubit) #add control-h_j (here h_j = ZZ)
        circuit_zz.cz(qubit_ancilla,qubit+1) #add control-h_j (here h_j = ZZ)
        circuit_zz.h(qubit_ancilla) #basis rotation for measurements in {|+>, |->} basis
        circuit_zz.measure(qubit_ancilla,0)
        circuit_dict.append(circuit_zz)
    return circuit_dict

def add_conditional_RX_gate(circuit,qubit_ancilla, qubit_target, param, params_index,deriv_index_list):
    ''' 
    if params_index is equal to one element in deriv_index_list, then add RZ.CZ to circuit (instead of RZ)
    - input:
        - circuit: quantum circuit
        - qubit_ancilla: index of ancilla qubit
        - qubit_target: index of target qubit (on which RZ acts on)
        - param: rotation angle
        - params_index: index of parameter
        - deriv_index_list: list of index [i,j] wrt which we take derivative (i<j)
    -output: None
    '''
    #derivative wrt RZ gate
    if deriv_index_list[0] == params_index or deriv_index_list[1] == params_index:
        circuit.x(qubit_ancilla) #add X gate
        circuit.cx(qubit_ancilla,qubit_target) #add CX gate 
    circuit.rx(param,qubit_target) #add RX gate
    
def add_conditional_RZZ_gate(circuit,qubit_ancilla, qubits_target, param, params_index,deriv_index_list):
    ''' 
    if params_index is equal to one element in deriv_index_list, then add RZ.CZ to circuit (instead of RZ) two between CNOTs
    - input:
        - circuit: quantum circuit
        - qubit_ancilla: index of ancilla qubit
        - qubits_target: list of indices of target qubits (on which RZ acts on)
        - param: rotation angle
        - params_index: index of parameter
        - deriv_index_list: list of index [i,j] wrt which we take derivative (i<j)
    -output: None
    '''
    #derivative wrt RZ gate
    circuit.cx(qubits_target[0],qubits_target[1])
    if deriv_index_list[0] == params_index or deriv_index_list[1] == params_index:
        circuit.x(qubit_ancilla) #add X gate
        circuit.cz(qubit_ancilla,qubits_target[1]) #add CZ gate 
    circuit.rz(param,qubits_target[1]) #add RZ gate
    circuit.cx(qubits_target[0],qubits_target[1])
    
def make_correction_circuit_HVA(params,num_qubits,num_layers,deriv_index):
    '''
    circuit for gamma_{i}
    - input
        - num_qubits: number of qubits (N)
        - num_layers: number of layers (L)
        - params: list parameters (\theta) (number of parameters: (3*num_qubits-2)*num_layers)
        - deriv_index: index of a parameter with respect to which we want to take derivative. (i for V_{i})

    '''
    circuit = QuantumCircuit(num_qubits+1,1) #N-register + ancilla
    qubit_ancilla = num_qubits #ancilla in last qubit

    # for qubit in range(num_qubits):
    #     circuit.add_H_gate(qubit)
    circuit.h(qubit_ancilla) # make (|0>+|1>)/sqrt(2) in ancilla qubit # fixed Nov. 29
    
    deriv_index_list = [deriv_index,len(params)*10]
    params_index = 0
    for layer in range(num_layers):
        #RZZ even
        for qubit in range(0,num_qubits-1,2):
            add_conditional_RZZ_gate(circuit,qubit_ancilla, [qubit,qubit+1], params[params_index], params_index,deriv_index_list)
            # add_RZZ_gate(circuit,[qubit,qubit+1],params[params_index])
            params_index += 1
        #RZZ odd
        for qubit in range(1,num_qubits-1,2):
            add_conditional_RZZ_gate(circuit,qubit_ancilla, [qubit,qubit+1], params[params_index], params_index,deriv_index_list)
            # add_RZZ_gate(circuit,[qubit,qubit+1],params[params_index])
            params_index += 1
        #RX
        for qubit in range(num_qubits):
            add_conditional_RX_gate(circuit,qubit_ancilla, qubit, params[params_index], params_index,deriv_index_list)
            # circuit.add_RZ_gate(qubit,params[params_index])
            params_index += 1

    circuit.x(qubit_ancilla) #add X gate
    circuit.h(qubit_ancilla) #basis rotation for measurements in {|+>, |->} basis
    circuit.measure(qubit_ancilla,0)
    return circuit

def compute_M_matrix_HVA(params,num_qubits,num_layers, sampler,nShots=100000):
    '''
    compute M_{ij}
    - input
        - num_qubits: number of qubits (N)
        - num_layers: number of layers (L)
        - params: list parameters (\theta) (number of parameters: 2*num_qubits*num_layers)
    - output: M_{ij} as numpy array
    '''
    num_params = len(params)
    M_mat = np.zeros((num_params,num_params)) # as numpy array
    qubit_ancilla = num_qubits
    for i in range(num_params-1):
        for j in range(i+1,num_params):
            deriv_index_list = [i,j]
            
            circuit_M = make_M_circuit_HVA(params,num_qubits,num_layers,deriv_index_list)
            circuit_corr1 = make_correction_circuit_HVA(params,num_qubits,num_layers,deriv_index_list[0])
            circuit_corr2 = make_correction_circuit_HVA(params,num_qubits,num_layers,deriv_index_list[1])
            
            job=sampler.run([circuit_M , circuit_corr1, circuit_corr2],shots=nShots)
            result=job.result()
            
            zero_prob = result.quasi_dists[0][0] 
            zero_prob_corr1 = result.quasi_dists[1][0] 
            zero_prob_corr2 = result.quasi_dists[2][0] 
            gamma1 = (2*zero_prob_corr1 - 1)
            gamma2 = (2*zero_prob_corr2 - 1)
            
            M_mat_element = (2*zero_prob- 1)/2 - gamma1*gamma2/2
            M_mat[i,j] = M_mat_element
            
    for i in range(num_params):
        circuit_correction = make_correction_circuit_HVA(params,num_qubits,num_layers,i)
        job=sampler.run([circuit_correction],shots=nShots)
        result=job.result()
        zero_probability_correction = result.quasi_dists[0][0] 
        gamma = (2*zero_probability_correction - 1)
        M_mat[i,i] = 1/2 - (gamma**2)/2 
        
    for i in range(num_params):
        for j in range(i):
            M_mat[i,j] = M_mat[j,i]
    return M_mat

def make_coeff_dict(num_qubits,j_coupling,h_coupling):
    '''
    make coefficient dictionary from Hamiltonian
    - input:
        - num_qubits: number of qubits (N)
        - j_coupling: coefficient of ZZ terms (J)
        - h_coupling: coefficient of X terms (h)
    - output: coefficient dictionary
        - key: terms in Hamiltonian (str)
        - value: coresponding coefficient
    '''
    # w = 1/2/a #coeff in front of fermion kin. term
    # j_coupling = g*g*a/2 #coeff in front of gauge kin. term

    #make output dictionary 
    #coeff_dict = {}
    coeff_dict = []

    #X terms
    for i in range(num_qubits):
        #term = f"X {i}"
        coeff_dict.append( -h_coupling)
    
    #ZZ terms
    for i in range(num_qubits-1):
        #term = f"Z {i} Z {i+1}"
        coeff_dict.append(-j_coupling)

    return coeff_dict

def compute_energy(params,num_qubits,num_layers,j_coupling,h_coupling,estimator) :

    ansatz = make_ansatz_HVA(params,num_qubits,num_layers)
    observable=SparsePauliOp.from_list([("X"+"I"*(num_qubits-1) , -h_coupling )])
    
    for i in np.arange(1,num_qubits,1):
        observable=observable + SparsePauliOp.from_list([("I"*(i)+"X"+"I"*(num_qubits-1-i) , -h_coupling )])
        
    for i in np.arange(0,num_qubits-1,1):
        observable=observable + (SparsePauliOp.from_list([("I"*(i)+"ZZ"+"I"*(num_qubits-2-i) ,-j_coupling)]))

    
    job=estimator.run(circuits=[ansatz], observables=[observable]
                          )


    return job.result().values[0]

def compute_energyGrad(params,num_qubits,num_layers,j_coupling,h_coupling,estimator) :

    ansatz = make_ansatz_HVA_weights(len(params),num_qubits,num_layers)
    observable=SparsePauliOp.from_list([("X"+"I"*(num_qubits-1) , -h_coupling )])
    
    for i in np.arange(1,num_qubits,1):
        observable=observable + SparsePauliOp.from_list([("I"*(i)+"X"+"I"*(num_qubits-1-i) , -h_coupling )])
        
    for i in np.arange(0,num_qubits-1,1):
        observable=observable + (SparsePauliOp.from_list([("I"*(i)+"ZZ"+"I"*(num_qubits-2-i) ,-j_coupling)]))

    gradEstimator= ParamShiftEstimatorGradient(estimator )
    gradJob=gradEstimator.run(circuits=[ansatz], observables=[observable]
                          , parameter_values=[params])

    return gradJob.result().gradients[0]

def compute_V_vector_HVA(params,num_qubits,num_layers,j_coupling,h_coupling,sampler, 
                         estimator, nShots=100000):
    '''
    compute V_{i}
    - input:
        - num_qubits: number of qubits (N)
        - j_coupling,h_coupling: Hamiltonian parameter (J, h)
    - output: V_{i} as numpy array
    '''
    num_params = len(params)
    V_vec = np.zeros(num_params) # as numpy array
    qubit_ancilla = num_qubits
    for i in range(num_params):

        deriv_index = i
        circuit_V = make_V_circuit_HVA(params,num_qubits,num_layers,deriv_index)
        job_V=sampler.run(circuit_V ,shots=nShots)
        job_V_results=job_V.result()
        coeff_dict = make_coeff_dict(num_qubits,j_coupling,h_coupling)
        V_vec_element = 0.0
        for j in range(len(circuit_V)):
            zero_probability = job_V_results.quasi_dists[j][0]
            V_vec_element += -(2*zero_probability - 1) * coeff_dict[j] #changed Dec.12, 2022

        #compute gamma_i
        circuit_correction = make_correction_circuit_HVA(params,num_qubits,num_layers,deriv_index)
        job_c=sampler.run(circuit_correction,shots=nShots)
        job_c_results=job_c.result()
        zero_probability_correction = job_c_results.quasi_dists[0][0]
        gamma = 2*zero_probability_correction - 1
        
        #compute <H>
        energy = compute_energy(params,num_qubits,num_layers,j_coupling,h_coupling,estimator)
        
        V_vec[i] = V_vec_element + gamma*energy
    return V_vec

def compute_params_list(params_init,num_qubits,num_layers,j_coupling,h_coupling,num_steps,
                        time_max,epsilon_det,sampler,estimator,nShots=1000,energy_correction=False,
                       alpha=0.1):
    ''' 
    compute evolution of parameters via VQS
    input:
        - params_init: initial parameters 
        - num_qubits: number of qubits
        - num_layers: number of layers
        - j_coupling: coupling (J)
        - h_coupling: coupling (h)
        - num_steps: number of time steps
        - time_max: final time
        - epsilon_det: regularization parameter (determinant cutoff)
    output: list of parameters
    '''
    params_list = [params_init] #list of parameters
    E_init=compute_energy(params_init,num_qubits,num_layers,j_coupling,h_coupling,estimator)
    energy_list=[E_init]
    delta_time = time_max/num_steps
    num_params = len(params_init)
    for step in range(1,num_steps):
        params = params_list[step-1]
        M_mat = compute_M_matrix_HVA(params,num_qubits,num_layers,sampler,nShots=nShots)
        V_vec = compute_V_vector_HVA(params,num_qubits,num_layers,j_coupling,h_coupling,sampler,estimator,nShots=nShots)
        M_mat_det = np.linalg.det(M_mat) #determinant of M matrix
        
        #(naive) regularization
        if M_mat_det <= epsilon_det:
            M_mat += np.identity(num_params)*epsilon_det #M <- M + eplsilon*I
        M_mat_inv = np.linalg.inv(M_mat)
        delta_params = M_mat_inv @ V_vec
        
        params_new = params + delta_params * delta_time
        params_list.append(params_new)
        energy = compute_energy(params_new,num_qubits,num_layers,j_coupling,h_coupling,estimator)
        energy_list.append(energy)
        
        if energy_correction:
            diff=(energy-E_init)
            grad=compute_energyGrad(params_new,num_qubits,num_layers,j_coupling,h_coupling,estimator) 
            params_new-=alpha*grad*diff

        if step % 10 == 0:
            print(f'step {step}: DONE')
    return np.array(params_list),np.array(energy_list)

def make_magnetization_op(num_qubits):
    '''
    input: num_qubits (number of qubits)
    output: magnetization operator as 'observable'
    '''
    observable=SparsePauliOp.from_list([("Z"+"I"*(num_qubits-1) , 1/num_qubits)])
    
    for i in np.arange(1,num_qubits,1):
        observable=observable + SparsePauliOp.from_list([("I"*i+"Z"+"I"*(num_qubits-i-1) , 1/num_qubits)])
        
    return observable

def compute_magnetization(params,num_qubits,num_layers,estimator):
    ''' 
    compute the expectation value of magnetization
    input: 
        - parmas: list of parmeters
        - num_qubits: number of qubits
        - num_layers: number of layers
    output: value of magnetization
    '''
    ansatz = make_ansatz_HVA(params,num_qubits,num_layers)
    
    observable = make_magnetization_op(num_qubits)
    
    job=estimator.run(circuits=[ansatz], observables=[observable]
                          )




    return job.result().values[0]