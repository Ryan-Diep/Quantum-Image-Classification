import pennylane as qml
from pennylane import numpy as np
from qiskit_machine_learning.utils import algorithm_globals
import matplotlib.pyplot as plt
import os

def preprocess_image(data):
    img = np.load(data)
    flattened_image = img.flatten()
    return flattened_image / np.linalg.norm(flattened_image)

def amplitude_encode(dataset_dir = "quantum_tetris_dataset"):
    dev = qml.device("default.qubit", wires=4) 

    @qml.qnode(dev)
    def circuit(image_data):
        qml.AmplitudeEmbedding(features=image_data, wires=range(4), normalize=False)
        return qml.probs(wires=range(4))  

    # all_probs = np.zeros((1000, 16))
    images = []
    labels = []

    os.chdir(dataset_dir)

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
                # all_probs[i] = circuit(normalized)
                images.append(image_angles)
                labels.append(label)
                
                #print(f"Processed {filename} ({i+1}/1000)")
                
            except FileNotFoundError:
                print(f"Missing {filename}, using zeros")
                random_data = algorithm_globals.random.uniform(0, np.pi/4, size=16)
                images.append(random_data)
                labels.append(0)
                # all_probs[i] = np.zeros(16)
                # labels.append('missing')

    os.chdir('..')
    # np.save("all_tetris_probs.npy", all_probs)
    np.save("all_tetris_probs.npy", np.array(images))
    np.save("tetris_labels.npy", np.array(labels))

    return np.array(images), np.array(labels)

amplitude_encode("quantum_tetris_dataset")