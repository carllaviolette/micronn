import numpy as np


# Inherit Layer Class
class Layer:

    def __init__(self):
        self.input = None
        self.output = None

    # Computes the output Y of a layer for a given input X.
    def forward(self, input_data):
        raise NotImplementedError

    # Computes dL/dX for a given dL/dY (and updates parameters if any).
    def backward(self, grad_output, learning_rate):
        raise NotImplementedError


# Linear Layer Class
class Linear(Layer):

  def __init__(self, nin, nout):
    self.layer_type = 'dense'
    self.weights = np.random.randn(nin, nout)
    self.biases = np.random.randn(1, nout)

  # Returns a linear transformation of the input: Y = WX + B
  def forward(self, input_data):
    self.input = input_data # X
    self.output = np.dot(self.input, self.weights) + self.biases
    return self.output

  # Given dL/dY (grad_output), computes the partial derivatives dL/dW (grad_weights), dL/dB (grad_biases) and dL/dX (grad_input) using the Chain Rule.
  def backward(self, grad_local, learning_rate):
    self.grad_output = grad_local
    self.grad_input = np.dot(self.grad_output, self.weights.T)
    self.grad_weights = np.dot(self.input.T, self.grad_output)
    self.grad_biases = self.grad_output

    # Update parameters.
    self.weights += -learning_rate * self.grad_weights
    self.biases += -learning_rate * self.grad_biases

    return self.grad_input


# ReLU Layer Class
class Relu(Layer):

  def __init__(self):
    self.layer_type = 'activation'

  # Returns the activated input: Y = ReLU(X).
  def forward(self, input_data):
    self.input = input_data
    self.output = np.maximum(0, self.input)
    return self.output

  # Given dL/dY (grad_output), computes the partial derivative dL/dX (grad_input) using the Chain Rule.
  # Note: learning_rate is included for consistency but not used since there are no "learnable" parameters in this activation layer.
  def backward(self, grad_local, learning_rate):
    self.grad_output = grad_local
    self.grad_input = self.grad_output * ((self.input >= 0) * 1)
    return self.grad_input


# Sigmoid Layer Class
class Sigmoid(Layer):

  def __init__(self):
    self.layer_type = 'activation'

  # Returns the activated input: Y = Sigmoid(X)
  def forward(self, input_data):
    self.input = input_data
    self.output = 1 / (1 + np.exp(self.input))
    return self.output

  # Given dL/dY (grad_output), computes the partial derivative dL/dX (grad_input) using the Chain Rule.
  # Note: learning_rate is included for consistency but not used since there are no "learnable" parameters in this activation layer.
  def backward(self, grad_local, learning_rate):
    self.grad_output = grad_output
    sigmoid = 1 / (1 + np.exp(self.input))
    self.grad_input = self.grad_output * (sigmoid * (1 - sigmoid))
    return self.grad_input


# Softmax Layer Class
class Softmax(Layer):

  def __init__(self):
    self.layer_type = 'activation'

  # Returns the activated input.
  def forward(self, input_data):
    self.input = input_data
    exp_norm = np.exp(self.input - np.max(self.input, axis=1, keepdims=True))
    self.output = exp_norm / np.sum(exp_norm, axis=1, keepdims=True)
    return self.output

  # Returns loss_input = dL/dX for a given loss_output = dL/dY.
  # learning_rate is not used because there are no "learnable" parameters in an activation layer.
  def backward(self, grad_local, learning_rate):
    self.grad_output = grad_local
    self.I = np.eye(self.input.size)
    self.jacobian = self.output * (self.I - self.output.T)
    self.grad_input = np.dot(self.grad_output, self.jacobian)
    return self.grad_input


# Tanh Layer Class
class Tanh(Layer):

  def __init__(self):
    self.layer_type = 'activation'

  # Returns the activated input: Y = Tanh(X).
  def forward(self, input_data):
    self.input = input_data
    self.output = np.tanh(self.input)
    return self.output

  # Given dL/dY (grad_output), computes the partial derivative dL/dX (grad_input) using the Chain Rule.
  # Note: learning_rate is included for consistency but not used since there are no "learnable" parameters in this activation layer.
  def backward(self, grad_local, learning_rate):
    self.grad_output = grad_local
    self.grad_input = self.grad_output * (1 - np.tanh(self.input)**2)
    return self.grad_input


# Loss Functions and Gradients
def mse(y_true, y_pred):
  loss = np.mean(np.power(y_true-y_pred, 2))
  return loss

def mse_grad(y_true, y_pred):
  grad_loss = 2*(y_pred-y_true)/y_true.size
  return grad_loss

def cross_entropy(y_true, y_pred):
  y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)
  loss = -np.dot(y_true, np.log(y_pred_clip).T)
  return loss

def cross_entropy_grad(y_true, y_pred):
  y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)
  grad_loss = -(y_true / y_pred_clip)
  return grad_loss


# Neural Network Class
class NeuralNetwork:

  def __init__(self):
    self.layers = []
    self.loss = None
    self.grad_loss = None

  # Define add layer method.
  def add(self, layer):
    self.layers.append(layer)

  # Define loss method.
  def loss_type(self, loss, grad_loss):
    self.loss = loss
    self.grad_loss = grad_loss

  # Define predict method.
  def predict(self, input_data):
    # sample dimension first
    samples = len(input_data)
    result = []

    # Run network over all samples.
    for i in range(samples):
      # Forward propagation.
      output = input_data[i]
      for layer in self.layers:
        output = layer.forward(output)
      result.append(output)

    return result

  # Define training method using standard Stochastic Gradient Descent.
  def train(self, x_train, y_train, epochs, learning_rate):

    samples = len(x_train)

    # Training loop.
    for i in range(epochs):
      total_loss = 0
      accuracy = 0
      for j in range(samples):
        # Forward propagation.
        output = x_train[j]
        for layer in self.layers:
          output = layer.forward(output)

        # Compute loss (for display purposes only).
        total_loss += self.loss(y_train[j], output)

        # Backward propagation.
        grad_local = self.grad_loss(y_train[j], output)
        for layer in reversed(self.layers):
          grad_local = layer.backward(grad_local, learning_rate)

        # Compute accuracy.
        sample_y_pred = np.array(output)
        sample_y_pred_max = np.argmax(sample_y_pred, axis=1)
        sample_y_true = np.array(y_train[j])
        sample_y_true_max = np.argmax(sample_y_true, axis=1)
        sample_accuracy = np.sum(sample_y_pred_max == sample_y_true_max) / 1
        accuracy += sample_accuracy

      # Calculate average error over all samples (one complete epoch).
      total_loss /= samples
      accuracy /= samples
      print('epoch %d/%d   loss=%f   accuracy=%f' % (i+1, epochs, total_loss, accuracy))