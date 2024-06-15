import math

# Set initial values
x = 1  # Input
w = 0.4  # Weight
bias = 1  # Bias
learning_rate = 0.1  # Learning rate
original_output = 0  # Desired output

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Derivative of sigmoid function
def sigmoid_derivative(output):
    return output * (1 - output)

# Forward Pass
z = x * w + bias  # Weighted sum
print(f"Weighted sum {z}")

nn_output = sigmoid(z)  # Activation function
print(f"Activation function {nn_output}")

# Loss calculation (Mean Squared Error)
loss = 0.5 * (nn_output - original_output) ** 2
print(f"Loss {loss}")

# Backward Pass
# Gradient of the loss with respect to the NN output
d_loss_d_output = nn_output - original_output
print(f"d_loss_d_output {d_loss_d_output}")

# Gradient of the NN output with respect to z
d_output_d_z = sigmoid_derivative(nn_output)
print(f"d_output_d_z {d_output_d_z}")

# Gradient of z with respect to w and bias
d_z_d_w = x  # Because z = x * w + bias, the partial derivative of z with respect to w is x
d_z_d_bias = 1  # Because z = x * w + bias, the partial derivative of z with respect to bias is 1

# Combine gradients to get the gradient of the loss with respect to w and bias
d_loss_d_w = d_loss_d_output * d_output_d_z * d_z_d_w
print(f"d_loss_d_w {d_loss_d_w}")
d_loss_d_bias = d_loss_d_output * d_output_d_z * d_z_d_bias
print(f"d_loss_d_bias {d_loss_d_bias}")

# Update weight and bias
w -= learning_rate * d_loss_d_w
bias -= learning_rate * d_loss_d_bias

# Print updated parameters
print(f"Updated w: {w}")
print(f"Updated bias: {bias}")