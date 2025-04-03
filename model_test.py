import json
import pickle
import numpy as np
from qiskit import QuantumCircuit
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator

with open("onehot_encoder.pkl", "rb") as f:
    enc = pickle.load(f)

def classify_new_image(new_image):
    loaded_classifier = NeuralNetworkClassifier.load('tetris_classifier.model')

    prediction = loaded_classifier.predict(new_image.reshape(1, -1))
    
    predicted_label = enc.inverse_transform(prediction)[0][0]
    
    return predicted_label

new_image = np.load('./test_quantum_tetris_dataset/tetris_0024.npy')
predicted_class = classify_new_image(new_image)
print(f"The image is classified as: {predicted_class}")