# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np


class Network(object):

    def __init__(self, sizes):
        """
        This states how the object ``Network`` is defined.
        
        The initializer takes as argument a list (``sizes``) containing
        the number of neurons in the respective layers of the network.
        
        For example, if the list was [2, 3, 1] then it would be a
        three-layer network, with the first layer containing 2 neurons,
        the second layer 3 neurons, and the third layer 1 neuron.
        
        The biases and weights for the network are initialized randomly,
        using a Gaussian distribution with mean 0, and variance 1.
        
        NOTE: the first layer is assumed to be an input layer,
        and by convention we won't set any biases for those neurons,
        since biases are only ever used in computing the outputs from
        later layers.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data):
        """
        Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory. If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        """

        # data to be used to train the network
        training_data = list(training_data)
        n = len(training_data)
        
        # data to check performance at the end of each epoch
        test_data = list(test_data)
        n_test = len(test_data)
        
        
        evaluation_accuracy = []
        training_accuracy = []
        for j in range(epochs):
            # shuffle to avoid effects of undesired correlations
            random.shuffle(training_data)
            
            # split data into ``mini_batches``
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            # each of them is used to estimate the gradient of the 
            # cost function in order to perform stochastic gradient
            # descent (SGD)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            correct = self.accuracy(test_data)
            print("Epoch {} : {} / {}".format(j,correct,n_test));
            evaluation_accuracy.append(correct/n_test)
            
            correct = self.accuracy(training_data, convert=True)
            training_accuracy.append(correct/n)
            
        return training_accuracy, evaluation_accuracy
    
    

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate.
        """
        
        # define the gradient vectors (with respect to the biases and
        # the weights)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # use the minibatch (passed as argument) to evaluate the 
        # gradients through backpropagation (see function `backprop`
        # below)
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        # change the weights of the `Network` object
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        This function is the core of the learning algorithm: backpropagation
        
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.
        ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of
        numpy arrays, similar to ``self.biases`` and ``self.weights``.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        #
        # 1. feedforward: calculate inputs and activities at each layer
        #
        activation = x    # this is the input layer (passed as argument)
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        #
        # 2. backward pass: calculate the ``error`` at each layer
        #      by propagating errors back
        #
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # Note that the numeration of layers, here grows from the output
        # to the input layer (Python convention for negative indices in
        # lists)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations.
        """
        return (output_activations-y)


    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy
    

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
