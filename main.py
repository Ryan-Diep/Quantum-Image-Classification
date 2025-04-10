import json
import matplotlib.pyplot as plt
import numpy as np
import pickle
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from amplitudeEmbedding import amplitude_encode

algorithm_globals.random_seed = 482988
estimator = Estimator()

# We now define a two qubit unitary as defined in [3]
def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target

# Let's draw this circuit and see what it looks like
params = ParameterVector("θ", length=3)
circuit = conv_circuit(params)
# circuit.draw("mpl", style="clifford")

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


circuit = conv_layer(4, "θ")

def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)

    return target


params = ParameterVector("θ", length=3)
circuit = pool_circuit(params)

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

def generate_tetris_dataset(num_samples_per_class=200):
    """Generate dataset of Tetris blocks with 5 different classes"""
    block_shapes = {
        'I': [(0,1), (1,1), (2,1), (3,1)],  # I piece
        'O': [(0,0), (0,1), (1,0), (1,1)],  # O piece (square)
        'T': [(0,1), (1,0), (1,1), (1,2)],  # T piece
        'L': [(0,0), (1,0), (2,0), (2,1)],  # L piece
        'S': [(0,1), (0,2), (1,0), (1,1)],  # S piece
    }
    
    images = []
    labels = []
    
    for block_type, coords in block_shapes.items():
        for _ in range(num_samples_per_class):
            image = np.zeros((4, 4))
            
            for x, y in coords:
                image[x, y] = 1
            
            for i in range(4):
                for j in range(4):
                    if image[i, j] == 0:
                        image[i, j] = algorithm_globals.random.uniform(0, 0.3)
            
            flat_image = image.flatten()
            encoded_image = np.array([val * np.pi/2 for val in flat_image])
            
            images.append(encoded_image)
            labels.append(block_type)
    
    return np.array(images), np.array(labels)

print("encode images")
images, labels = generate_tetris_dataset(100)

print("encoding done")

print("split data")
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.3, random_state=246
)
print("finished splitting")

feature_map = ZFeatureMap(16)

ansatz = QuantumCircuit(16, name="Ansatz")

# First Convolutional Layer
ansatz.compose(conv_layer(16, "c1"), list(range(16)), inplace=True)

# First Pooling Layer
ansatz.compose(pool_layer([0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], "p1"), list(range(16)), inplace=True)

# Second Convolutional Layer
ansatz.compose(conv_layer(8, "c2"), list(range(8, 16)), inplace=True)

# Second Pooling Layer
ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p2"), list(range(8, 16)), inplace=True)

# Third Convolutional Layer
ansatz.compose(conv_layer(4, "c3"), list(range(12, 16)), inplace=True)

# Third Pooling Layer
ansatz.compose(pool_layer([0, 1], [2, 3], "p3"), list(range(12, 16)), inplace=True)

# Fourth Convolutional Layer
ansatz.compose(conv_layer(2, "c4"), list(range(14, 16)), inplace=True)

# Fourth Pooling Layer
ansatz.compose(pool_layer([0], [1], "p4"), list(range(14, 16)), inplace=True)

# Combining the feature map and ansatz
circuit = QuantumCircuit(16)
circuit.compose(feature_map, range(16), inplace=True)
circuit.compose(ansatz, range(16), inplace=True)

print("adding observables")
observables = []
observables.append(SparsePauliOp.from_list([("I" * 14 + "ZI", 1)]))
observables.append(SparsePauliOp.from_list([("I" * 14 + "XI", 1)]))
observables.append(SparsePauliOp.from_list([("I" * 14 + "YI", 1)]))
observables.append(SparsePauliOp.from_list([("I" * 14 + "IZ", 1)]))
observables.append(SparsePauliOp.from_list([("I" * 14 + "IX", 1)]))
print("finished adding observables")

# we decompose the circuit for the QNN to avoid additional data copying
print("decompose circuit")
qnn = EstimatorQNN(
    circuit=circuit.decompose(),
    observables=observables,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    estimator=estimator,
)
print("finished decomposing circuit")

num_params = len(ansatz.parameters)

# Generate random  point
# print("generate random point")
# initial_point = np.random.uniform(-np.pi, np.pi, num_params)

# with open("11_qcnn_initial_point.json", "w") as f:
#     json.dump(initial_point.tolist(), f)

# with open("11_qcnn_initial_point.json", "r") as f:
#     initial_point = json.load(f)
# print("finished generating initial point")

with open("16_qcnn_trained_weights.json", "r") as f:
    initial_point = json.load(f)

def callback_graph(weights, obj_func_eval):
    print(f"Iteration {len(objective_func_vals)}, Objective: {obj_func_eval}")
    objective_func_vals.append(obj_func_eval)

# classifier properities 
classifier = NeuralNetworkClassifier(
    qnn,
    optimizer=COBYLA(maxiter=1000),  # Set max iterations here
    callback=callback_graph,
    initial_point=initial_point,
    one_hot=True
)

print("running training images")
x = np.asarray(train_images)
enc = OneHotEncoder(sparse_output=False)
y = enc.fit_transform(np.array(train_labels).reshape(-1, 1))
print("finished running training images")

# Manually verify output shape
test_output = qnn.forward(train_images[0], initial_point)
print(f"QNN output shape: {test_output.shape}")  # Should be (1, 8)

print("fit training data")
objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)
classifier.fit(x, y)
print("finished fitting training data")

# score classifier
print(f"Accuracy from the train data : {np.round(100 * classifier.score(x, y), 2)}%")

print("running test data") 
x = np.asarray(test_images)
y_test = enc.transform(np.array(test_labels).reshape(-1, 1)) 
test_accuracy = classifier.score(x, y_test)
print("finished fitting test data")

print(f"Accuracy from the test data : {np.round(100 * test_accuracy, 2)}%")

with open("onehot_encoder.pkl", "wb") as f:
    pickle.dump(enc, f)

with open("16_qcnn_trained_weights.json", "w") as f:
    json.dump(classifier.weights.tolist(), f)

classifier.save("tetris_classifier.model")