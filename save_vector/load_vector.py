import numpy as np

with open('app_vector.npy', 'rb') as f:
    embedding = np.loadtxt(f)

print(embedding.shape)
