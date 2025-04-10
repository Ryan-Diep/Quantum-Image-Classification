"""
We implemented amplitude encoding but this proved to be slow and usuable in our case

"""

import pennylane as qml
from pennylane import numpy as np
from qiskit_machine_learning.utils import algorithm_globals
import matplotlib.pyplot as plt
import os

# flatten image 
def preprocess_image(data):
    img = np.load(data)
    flattened_image = img.flatten()
    return flattened_image / np.linalg.norm(flattened_image)

def amplitude_encode(dataset_dir = "quantum_tetris_dataset"):
    # Create a quantum device with 4 qubits
    dev = qml.device("default.qubit", wires=4) 

    # Define a quantum circuit to encode image data using amplitude embedding
    @qml.qnode(dev)
    def circuit(image_data):
        # Embed the image data into the amplitude of a quantum state
        qml.AmplitudeEmbedding(features=image_data, wires=range(4), normalize=False)
        return qml.probs(wires=range(4))  

    images = []
    labels = []

    os.chdir(dataset_dir)
    
    # Open and read the labels file (this no longer works)
    with open('labels.txt', 'r') as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
                
            try:
                row = line.strip().split(',')
                filename = row[0]
                label = int(row[1])
                
                normalized = preprocess_image(filename)
                probs = circuit(normalized)
                image_angles = probs * (np.pi/2)

                for j in range(16):
                    if image_angles[j] == 0:
                        image_angles[j] = algorithm_globals.random.uniform(0, np.pi/4)

                images.append(image_angles)
                labels.append(label)
                
            except FileNotFoundError:
                print(f"Missing {filename}, using zeros")
                random_data = algorithm_globals.random.uniform(0, np.pi/4, size=16)
                images.append(random_data)
                labels.append(0)

    os.chdir('..')
    np.save("all_tetris_probs.npy", np.array(images))
    np.save("tetris_labels.npy", np.array(labels))

    return np.array(images), np.array(labels)