import numpy as np

# machine learning is a technique in which you train the system to solve a problem instead of explicitly programming the rules
class NeuralNetwork:

    # constructor
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()]) # set two random initial weights
        self.bias = np.random.randn() # set random initial bias
        self.learning_rate = learning_rate

    # sigmoid function: 1 / (1 + e^(-x))
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # derivative of sigmoid function
    def sigmoid_deriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias # layer one function (dot product)
        layer_2 = self.sigmoid(layer_1) # layer two function (sigmoid function)
        return layer_2 # end result
    
    # performs backtracking and returns result of derivative function composition
    def compute_gradients(self, input_vector, target):

        # initial prediction
        prediction = self.predict(input_vector)

        # relevant derivatives (backtracking)
        derror_dprediction = 2 * (prediction - target) # derivative of the error, which is prediction - target
        dprediction_dlayer1 = self.sigmoid_deriv(np.dot(input_vector, self.weights) + self.bias) # derivative of sigmoid function
        dlayer1_dbias = 1 # derivative of bias
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector) # derivative of weights
        derror_dbias = (derror_dprediction * dprediction_dlayer1 * dlayer1_dbias) # chain rule
        derror_dweights = (derror_dprediction * dprediction_dlayer1 * dlayer1_dweights) # chain rule
        return derror_dbias, derror_dweights

    # performs backpropogation using values determing during the previous backtracking step, updates bias and weights accordingly
    def update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (derror_dweights * self.learning_rate)

    # performs training of network using given input_vectors, targets, and number of iterations
    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):

            # pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index] # grab the random index entry from the input_vectors array
            target = targets[random_data_index] # grab the random index entry from the targets array

            # compute the gradients and update the weights using backtracking and backpropogation
            derror_dbias, derror_dweights = self.compute_gradients(input_vector, target) # backtracking
            self.update_parameters(derror_dbias, derror_dweights) # backpropogation

            # measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0

                # loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors