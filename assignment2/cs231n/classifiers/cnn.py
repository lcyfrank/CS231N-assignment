from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################

        self.params['W1'] = np.random.randn(num_filters, input_dim[0], filter_size, filter_size) * weight_scale
        w1_height = ((input_dim[1] - 2) // 2 + 1) * ((input_dim[2] - 2) // 2 + 1) * num_filters  # the size after pool layer
        self.params['W2'] = np.random.randn(w1_height, hidden_dim) * weight_scale
        self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale

        self.params['b1'] = np.zeros(num_filters)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        cache = {}
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        out, cache['conv'] = conv_forward_naive(X, W1, b1, conv_param)  # conv
        out, cache['relu1'] = relu_forward(out)  # relu
        out, cache['pool'] = max_pool_forward_naive(out, pool_param)  # pool
        out, cache['affine1'] = affine_forward(out, W2, b2)  # affine
        out, cache['relu2'] = relu_forward(out)  # relu
        scores, cache['affine2'] = affine_forward(out, W3, b3)  # affine

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        true_class_score = scores[np.arange(X.shape[0]), y]
        log_inner = np.exp(true_class_score) / np.exp(scores).sum(1)
        item_loss = -np.log(log_inner)
        loss = item_loss.sum() / X.shape[0]
        loss += np.sum(self.params['W2'] ** 2) * self.reg / 2
        loss += np.sum(self.params['W3'] ** 2) * self.reg / 2

        dscore = np.exp(scores) / np.exp(scores).sum(1)[:, np.newaxis]
        dscore[np.arange(scores.shape[0]), y] = log_inner - 1
        dscore /= scores.shape[0]

        dout, grads['W3'], grads['b3'] = affine_backward(dscore, cache['affine2'])
        grads['W3'] += W3 * self.reg
        dout = relu_backward(dout, cache['relu2'])
        dout, grads['W2'], grads['b2'] = affine_backward(dout, cache['affine1'])
        grads['W2'] += W2 * self.reg
        dout = max_pool_backward_naive(dout, cache['pool'])
        dout = relu_backward(dout, cache['relu1'])

        _, grads['W1'], grads['b1'] = conv_backward_naive(dout, cache['conv'])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
