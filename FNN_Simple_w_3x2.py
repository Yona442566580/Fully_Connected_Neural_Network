import numpy as np

# active function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 初始化数据
X = np.array([[0, 0, 0], [0,0, 1], [0, 1, 0], [0, 1, 1]])  # Input date to train
y = np.array([[0], [0.1], [0.2], [0.3]])  # Label

np.random.seed(42)
weights_input_hidden = np.random.rand(3, 2)  # Weight: Input layer to hidden layer
weights_hidden_output = np.random.rand(2, 1)  # Weight: hidden layer to output layer
bias_hidden = np.random.rand(1, 2)  # Bias of hidden layer
bias_output = np.random.rand(1, 1)  # Bias of output layer

# learning rate
learning_rate = 0.1

# Train
for epoch in range(10000):  # Epoch
    # Forward
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Error function
    error = 1 / 2 * (predicted_output - y) * (predicted_output - y)  # 对 predicted_output 求导(derivative)之后是  predicted_output - y

    # Backward
    #
    d_predicted_output = (predicted_output - y) * sigmoid_derivative(output_layer_input) #
    #
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T) # T
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_input)

    # Weight update
    weights_hidden_output -= hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output -= np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate

    weights_input_hidden -= X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden -= np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

print("weights_hidden_output",weights_hidden_output)
print("weights_input_hidden",weights_input_hidden)

# Output
print("Output after train：")
print(predicted_output)

