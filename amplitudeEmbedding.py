import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess_image(data):
    img = np.load(data)
    flattened_image = img.flatten()
    return flattened_image / np.linalg.norm(flattened_image)

dev = qml.device("default.qubit", wires=4) 

@qml.qnode(dev)
def circuit(image_data):
    qml.AmplitudeEmbedding(features=image_data, wires=range(4), normalize=False)
    return qml.probs(wires=range(4))  

all_probs = np.zeros((1000, 16))

dataset_dir = "quantum_tetris_dataset"

os.chdir(dataset_dir)

for i in range(1000):
    try:
        filename = f"tetris_{i:04d}.npy"
        
        normalized = preprocess_image(filename)
        
        all_probs[i] = circuit(normalized)
        
        print(f"Processed {filename} ({i+1}/1000)")
    
    except FileNotFoundError:
        print(f"Missing {filename}, using zeros")
        all_probs[i] = np.zeros(16) 

np.save("all_tetris_probs.npy", all_probs)
print("Saved all probabilities to all_tetris_probs.npy")

avg_probs = np.mean(all_probs, axis=0)
print("\nAverage probabilities per state:")
for j in range(16):
    print(f"|{j:04b}⟩: {avg_probs[j]:.4f}")