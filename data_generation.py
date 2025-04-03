import numpy as np
import matplotlib.pyplot as plt
import os
import random
from matplotlib.colors import LinearSegmentedColormap

folder_name = "test_quantum_tetris_dataset"

# Create directory for the dataset
os.makedirs(folder_name, exist_ok=True)

num_images = 150

# Define the Tetris pieces for 4x4 grid
# Each piece is represented as a list of (row, col) coordinates
tetris_pieces = {
    'I': [(0,1), (1,1), (2,1), (3,1)],  # I piece
    'O': [(0,0), (0,1), (1,0), (1,1)],  # O piece (square)
    'T': [(0,1), (1,0), (1,1), (1,2)],  # T piece
    'L': [(0,0), (1,0), (2,0), (2,1)],  # L piece
    'S': [(0,1), (0,2), (1,0), (1,1)],  # S piece
}

label_mapping = {
    'I': 0,
    'O': 1,
    'T': 2,
    'L': 3,
    'S': 4,
}

# Function to rotate a piece
def rotate_piece(piece, times=1):
    rotated = piece.copy()
    for _ in range(times):
        # For a 4x4 grid, the rotation pivot is (1.5, 1.5)
        new_rotated = []
        for r, c in rotated:
            # Rotation formula around (1.5, 1.5)
            new_r = int(round(1.5 - (c - 1.5)))
            new_c = int(round(1.5 + (r - 1.5)))
            new_rotated.append((new_r, new_c))
        rotated = new_rotated
    return rotated

# Function to create a random non-Tetris shape
def random_non_tetris_shape():
    # Create a shape with 3-5 blocks (different from standard Tetris pieces)
    num_blocks = random.randint(3, 5)
    shape = []
    
    # Start with a random position
    r, c = random.randint(0, 3), random.randint(0, 3)
    shape.append((r, c))
    
    # Add adjacent blocks randomly
    for _ in range(num_blocks - 1):
        if not shape:
            r, c = random.randint(0, 3), random.randint(0, 3)
            shape.append((r, c))
            continue
            
        # Pick a random existing block
        base_r, base_c = random.choice(shape)
        
        # Try to add adjacent blocks
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        
        for dr, dc in directions:
            new_r, new_c = base_r + dr, base_c + dc
            if 0 <= new_r < 4 and 0 <= new_c < 4 and (new_r, new_c) not in shape:
                shape.append((new_r, new_c))
                break
    
    return shape

# Function to generate a random image matrix
def generate_random_matrix(idx):
    # Create an empty 4x4 grid
    grid = np.zeros((4, 4))
    
    # Select a random Tetris piece
    piece_type = random.choice(list(tetris_pieces.keys()))
    piece = tetris_pieces[piece_type]
    
    # Randomly rotate the piece
    rotations = random.randint(0, 3)
    if rotations > 0:
        piece = rotate_piece(piece, rotations)
    
    # Randomly shift the piece within grid bounds
    shift_successful = False
    max_attempts = 10
    attempts = 0
    
    while not shift_successful and attempts < max_attempts:
        r_shift = random.randint(-3, 3)
        c_shift = random.randint(-3, 3)
        
        shifted_piece = [(r + r_shift, c + c_shift) for r, c in piece]
        
        # Check if the shifted piece is within bounds
        if all(0 <= r < 4 and 0 <= c < 4 for r, c in shifted_piece):
            piece = shifted_piece
            shift_successful = True
        
        attempts += 1
    
    label = piece_type[0]  # Use first letter as label

    
    # Set the values for the piece blocks (between 0.7 and 1.0)
    for r, c in piece:
        if 0 <= r < 4 and 0 <= c < 4:  # Ensure it's within bounds
            grid[r, c] = random.uniform(0.7, 1.0)
    
    # Set background pixel values (between 0 and 0.3)
    for r in range(4):
        for c in range(4):
            if grid[r, c] == 0:
                grid[r, c] = random.uniform(0, 0.3)
    
    # Add some noise to the image
    noise = np.random.uniform(-0.05, 0.05, (4, 4))
    grid = np.clip(grid + noise, 0, 1)
    
    return grid, label

# Create a custom colormap similar to the image
# Using actual RGB values instead of color names
colors = [(0.3, 0, 0.5), (0.5, 0.8, 0), (1, 1, 0)]  # Purple, lime green, yellow
cmap = LinearSegmentedColormap.from_list('tetris_cmap', colors, N=256)

# Generate and save num_images matrices
matrices = []
labels = []

for i in range(num_images):
    matrix, label = generate_random_matrix(i)
    matrices.append(matrix)
    labels.append(label)
    
    # Save matrix to numpy file
    np.save(f'{folder_name}/tetris_{i:04d}.npy', matrix)
    
    # Also save a visualization for reference
    plt.figure(figsize=(2, 2))
    plt.imshow(matrix, cmap=cmap)
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.savefig(f'{folder_name}/tetris_{i:04d}_viz.png')
    plt.close()

# Save all matrices and labels to a single file for easy loading
np.save(f'{folder_name}/all_matrices.npy', np.array(matrices))
np.save(f'{folder_name}/all_labels.npy', np.array(labels))

# Save labels to a text file as well
with open(f'{folder_name}/labels.txt', 'w') as f:
    for i, label in enumerate(labels):
        numerical_label = label_mapping[label]
        f.write(f'tetris_{i:04d}.npy,{numerical_label}\n')

print(f"Generated num_images matrices in the '{folder_name}' folder")

# Display a few examples
plt.figure(figsize=(15, 3))
for i in range(5):
    idx = random.randint(0, num_images - 1)
    plt.subplot(1, 5, i+1)
    plt.imshow(matrices[idx], cmap=cmap)
    plt.title(f'Label: {labels[idx]}')
    plt.axis('on')  # Turn axes on to see the grid
    plt.grid(True)  # Show grid
plt.tight_layout()
plt.show()

# Generate example data in format suitable for quantum encoding
print("\nExample matrix for quantum encoding:")
example = matrices[0]
print(example)
print(f"Shape: {example.shape}, Label: {labels[0]}")
print("Values range from 0-0.3 (background) to 0.7-1.0 (piece)")