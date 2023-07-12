from itertools import product
from decimal import Decimal

import numpy as np
from qiskit import *
from qiskit import QuantumCircuit

# Classes for building up a directed-acyclic graph (DAG) structure
from qiskit.circuit import QuantumRegister

# Need gate classes for generating the Pauli twirling sets
from qiskit.circuit.library import (CXGate, CZGate, ECRGate, IGate, XGate,
                                    YGate, ZGate)
from qiskit.dagcircuit import DAGCircuit


# Operator class
from qiskit.quantum_info import Operator

# Transpiler stuff neded to make a pass and passmanager
from qiskit.transpiler.basepasses import TransformationPass


# ZNE
class Local2qFolding(TransformationPass):
    def __init__(self, scale_factor, folding_gate):
        super().__init__()
        self.scale_factor = scale_factor
        self.folding_gate = folding_gate
    
    def _next_odd_if_not_odd(self, value: int | float) -> int:
        return int(np.ceil(value) // 2 * 2 + 1)
    
    def _get_2q_gate_indices_to_be_folded(self, num_2q_gates, scale_factor, odd_fold_factor):
        num_2q_gates_to_be_folded = int(np.floor(
            num_2q_gates * float(Decimal(str(scale_factor)) - Decimal(1)) / float(Decimal(str(odd_fold_factor)) - 1),
        ))
        
        return np.random.choice(
            a=num_2q_gates,
            size=num_2q_gates_to_be_folded,
            replace=False,
        )

    def run(self, dag):
        if self.scale_factor == 1:
            return dag
        
        num_2q_gates = dag.count_ops()[self.folding_gate]
        odd_fold_factor = self._next_odd_if_not_odd(self.scale_factor)
        folding_gate_indices = self._get_2q_gate_indices_to_be_folded(
            num_2q_gates=num_2q_gates,
            scale_factor=self.scale_factor,
            odd_fold_factor=odd_fold_factor,
        )

        idx = 0
        for run in dag.collect_runs([self.folding_gate]):
            for node in run:
                fold_factor = odd_fold_factor if idx in folding_gate_indices else 1
                
                fold_dag = DAGCircuit()
                # Add a register of qubits (here always 2Q)
                qreg = QuantumRegister(2)
                fold_dag.add_qreg(qreg)
                for _ in range(fold_factor):
                    fold_dag.apply_operation_back(node.op, [qreg[0], qreg[1]])
                dag.substitute_node_with_dag(node, fold_dag)
                
                idx += 1
                
        return dag
    

# PT
def generate_pauli_twirling_sets(two_qubit_gate):
    """Generate the Pauli twirling sets for a given 2Q gate
    
    Sets are ordered such that gate[0] and gate[1] are pre-roations
    applied to control and target, respectively.  gate[2] and gate[3]
    are post-rotations for control and target, respectively.
    
    Parameters:
        two_qubit_gate (Gate): Input two-qubit gate
        
    Returns:
        list: List of all twirling gate sets
    """
    # Single qubit Pauli gates
    I = IGate()
    Z = ZGate()
    X = XGate()
    Y = YGate()

    # 2Q entangling gates
    two_qubit_gate = {
        "cx": CXGate(),
        "cz": CZGate(),
        "ecr": ECRGate(),
    }[two_qubit_gate]
    # Generate 16 element list of Pauli gates, each repeated 4 times
    operator_list = [I, Z, X, Y]
    # This is the target unitary to which our twirled circuit should match
    target_unitary = Operator(two_qubit_gate.to_matrix())
    twirling_sets = []
    
    # For every combination in 16 choose 4 make a circuit and look for equivilence
    for gates in product(operator_list, repeat=4):
        # Build a circuit for our twirled 2Q gate
        qc = QuantumCircuit(2)
        qc.append(gates[0], [0])
        qc.append(gates[1], [1])
        qc.append(two_qubit_gate, [0, 1])
        qc.append(gates[2], [0])
        qc.append(gates[3], [1])
        
        norm = np.linalg.norm(Operator.from_circuit(qc)-target_unitary)
        
        phase = None
        # If unitaries match we have a phase of zero
        if abs(norm) < 1e-15:
            phase = 0
        # If unitaries differ by a phase of pi, shift by pi
        elif abs(norm-4) < 1e-15:
            phase = np.pi

        if phase is not None:
            qc.global_phase += phase
            # Verify that our twirled circuit is a valid replacement
            assert Operator.from_circuit(qc) == target_unitary
            twirl_set = (gates, phase)
            # Check that set does not already exist
            if twirl_set not in twirling_sets:
                twirling_sets.append(twirl_set)
            
    return twirling_sets


class PauliTwirling(TransformationPass):
    """Pauli twirl an input circuit.
    """
    def __init__(self, twirling_gate, seed=None):
        """
        Parameters:
            twirling_gate (str): Which gate to twirl
            seed (int): Seed for RNG, should be < 2e32
        """
        super().__init__()
        # This is the target gate to twirl
        self.twirling_gate = twirling_gate
        # Get the twirling set from the dict we generated above
        # This should be repalced by a cached version in practice
        self.twirling_set = generate_pauli_twirling_sets(
            two_qubit_gate=twirling_gate
        )
        # Length of the twirling set to bound RNG generation
        self.twirling_len = len(self.twirling_set)
        # Seed the NumPy RNG
        self.rng = np.random.default_rng(seed)

    def run(self, dag):
        """Insert Pauli twirls into input DAG
        
        Parameters:
            dag (DAGCircuit): Input DAG
        
        Returns:
            dag: DAG with twirls added in-place
        """
        for run in dag.collect_runs([self.twirling_gate]):
            for node in run:
                # Generate a random int to specify the twirling gates
                twirl_idx = self.rng.integers(0, self.twirling_len)
                # Get the randomly selected twirling set
                twirl_gates = self.twirling_set[twirl_idx][0]
                twirl_phase = self.twirling_set[twirl_idx][1]
                # Make a small DAG for the twirled circuit we are going to insert
                twirl_dag = DAGCircuit()
                # Add a register of qubits (here always 2Q)
                qreg = QuantumRegister(2)
                twirl_dag.add_qreg(qreg)
                # gate[0] pre-applied to control
                twirl_dag.apply_operation_back(twirl_gates[0], [qreg[0]])
                # gate[1] pre-applied to target
                twirl_dag.apply_operation_back(twirl_gates[1], [qreg[1]])
                # Insert original gate
                twirl_dag.apply_operation_back(node.op, [qreg[0], qreg[1]])
                # gate[2] pre-applied to control
                twirl_dag.apply_operation_back(twirl_gates[2], [qreg[0]])
                # gate[3] pre-applied to target
                twirl_dag.apply_operation_back(twirl_gates[3], [qreg[1]])
                # Add a global phase gate to account for possible phase difference
                twirl_dag.global_phase += twirl_phase
                # Replace the target gate with the twirled version
                dag.substitute_node_with_dag(node, twirl_dag)
        
        return dag

def aggregate_data(data, normalize=False):
    aggregated = {}

    total = len(data) if normalize else 1
    
    for dist in data:
        for key in dist:
            aggregated[key] = aggregated.get(key, 0) + (dist[key]/total)
            
    return aggregated
