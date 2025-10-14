
import numpy as np
from sklearn.neural_network import MLPClassifier

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])  

model = MLPClassifier(
    hidden_layer_sizes=(2,), 
    activation='tanh',      
    solver='lbfgs',          
    max_iter=1000,
    random_state=42
)

model.fit(X, y)

predictions = model.predict(X)
accuracy = model.score(X, y)

print("===== XOR Problem using MLP =====")
print("Input:\n", X)
print("Expected Output:", y)
print("Predicted Output:", predictions)
print(f"Accuracy: {accuracy*100:.2f}%")
print("=================================")