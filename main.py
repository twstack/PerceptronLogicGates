from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# Creating and visualizing AND Data
data = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels = [0, 0, 0, 1]

plt.scatter([point[0] for point in data], [point[1] for point in data], c=labels)
plt.title("AND gate")
plt.show()

# Building the Perceptron
classifier = Perceptron(max_iter=40)
classifier.fit(data, labels)

# Changing the labels to represent XOR gate
labels = [0, 1, 1, 0]
plt.scatter([point[0] for point in data], [point[1] for point in data], c=labels)
plt.title("XOR gate")
plt.show()

# Changing the labels to represent OR gate
labels = [0, 1, 1, 1]
plt.scatter([point[0] for point in data], [point[1] for point in data], c=labels)
plt.title("OR gate")
plt.show()

# Visualizing the Perceptron
labels = [0, 0, 0, 1]  # Reset to AND gate

x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)

point_grid = list(product(x_values, y_values))

distances = classifier.decision_function(point_grid)
abs_distances = [abs(pt) for pt in distances]
distances_matrix = np.reshape(abs_distances, (100, 100))
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)
plt.scatter([point[0] for point in data], [point[1] for point in data], c=labels)
plt.colorbar(heatmap)
plt.title("AND gate Perceptron Visualization (Distance from Perceptron)\n")
plt.show()

# Change labels to XOR and retrain
labels = [0, 1, 1, 0]
classifier.fit(data, labels)

# Calculate distances and plot heatmap
distances = classifier.decision_function(point_grid)
abs_distances = [abs(pt) for pt in distances]
distances_matrix = np.reshape(abs_distances, (100, 100))
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)
plt.scatter([point[0] for point in data], [point[1] for point in data], c=labels)
plt.colorbar(heatmap)
plt.title("XOR gate Perceptron Visualization (Distance from Perceptron)\n")
plt.show()
# -> XOR not linearly separable - so no decision boundary can be visualized!

# Change labels to OR and retrain
labels = [0, 1, 1, 1]
classifier.fit(data, labels)

# Calculate distances and plot heatmap
distances = classifier.decision_function(point_grid)
abs_distances = [abs(pt) for pt in distances]
distances_matrix = np.reshape(abs_distances, (100, 100))
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)
plt.scatter([point[0] for point in data], [point[1] for point in data], c=labels)
plt.colorbar(heatmap)
plt.title("OR gate Perceptron Visualization (Distance from Perceptron)\n")
plt.show()

