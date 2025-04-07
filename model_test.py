import json
import pickle
import numpy as np
from qiskit import QuantumCircuit
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator

with open("onehot_encoder.pkl", "rb") as f:
    enc = pickle.load(f)

classifier = NeuralNetworkClassifier.load("tetris_classifier.model")

# L
new_image = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 1, 1]
])

# O
# new_image = np.array([
#     [0, 0, 0, 0],
#     [0, 1, 1, 0],
#     [0, 1, 1, 0],
#     [0, 0, 0, 0]
# ])

# I
# new_image = np.array([
#     [0, 1, 0, 0],
#     [0, 1, 0, 0],
#     [0, 1, 0, 0],
#     [0, 1, 0, 0]
# ])

# S
# new_image = np.array([
#     [0, 0, 0, 0],
#     [1, 0, 0, 0],
#     [1, 1, 0, 0],
#     [0, 1, 0, 0]
# ])

# T (sometimes)
# new_image = np.array([
#     [0, 0, 0, 0],
#     [0, 0, 1, 0],
#     [0, 1, 1, 0],
#     [0, 0, 1, 0]
# ])

for i in range(4):
    for j in range(4):
        if new_image[i, j] == 0:
            new_image[i, j] = np.random.uniform(0, 0.3)

encoded_image = np.array([val * np.pi/2 for val in new_image.flatten()])
prediction = classifier.predict([encoded_image])
predicted_label = enc.inverse_transform(prediction)
print(f"Predicted block type: {predicted_label[0]}")