from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

# Path to the .mat file
file_path = './caltech101_silhouettes_28.mat'

# Load the .mat file
data = loadmat(file_path)
print(type(data["classnames"]))

index = 10
image = data["X"][index].reshape((28, 28)).T
label = data["Y"][0][index]

plt.imshow(image, cmap='gray')  # Use 'gray' colormap for grayscale
plt.title(f"28x28 Image with label {data["classnames"][0][label-1][0]} with class number {label}")
plt.axis('off')
plt.show()
