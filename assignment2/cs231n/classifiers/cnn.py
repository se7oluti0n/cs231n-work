import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class FirstConvnet(object):
  """
  { conv-[batchnorm]-relu- [dropout] - pool} x N - conv - [batchnorm] - relu - [dropout] - {affine - [batch norm] - relu - [dropout]} x (M - 1) - affine - [softmax or SVM]

  The network operates on minibatches of data that have shape (N,C,H,W)
  consisting of N images, each with height H and width W and with C of channels
  """

  def __init__(self, input_dim=(3,32,32), num_filters=[1], filter_sizes=[7],
               hidden_dims=[100], num_classes=10, use_batchnorm=False, dropout=0, 
               weight_scale=1e-3, reg=0.0, dtype=np.float32, seed=None):
    """
    Initialize the network

    Inputs:
      - input_dim: Tuple (C,H,W) giving size of input data
      - num_filters: Lists of interger giving the number of filters in each convolutional layer
      - filter_size: List of interger giving the filter size in each convolutional layers
      - hidden_dims: List of integer giving the size of each full-connected hidden layer
      - num_classes: Number of scores to produce the final affine layer
      - use_batchnorm: Whether or not using batchnorm
      - dropout: dropout strength. if dropout=0 the not use dropout at all 
      - weight_scale: Scalr giving the standard deviation for random initialization of weights
      - reg: Scalar giving L2 relularization strength
      - dtype: numpy datatype to use for computation
    """

    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_fullconnected = 1 + len(hidden_dims)
    self.num_conv = len(num_filters)
    self.num_layers = self.num_fullconnected + self.num_conv
    self.params = {}
    self.conv_params = []
    self.pool_params = []
    self.reg = reg

    
    num_filters = [input_dim[0]] + num_filters

    for i in xrange(1, len(num_filters)):
      self.params['W'+str(i)] = weight_scale * np.random.randn(num_filters[i], num_filters[i-1], filter_sizes[i-1], filter_sizes[i-1])
      self.params['b'+str(i)] = np.zeros(num_filters[i])
      self.conv_params.append({'stride': 1, 'pad': (filter_sizes[i-1] - 1) / 2})
      if (i < self.num_conv):
        self.pool_params.append({'stride' : 2, 'pool_width': 2, 'pool_height': 2})



    last_convolution_size = num_filters[-1] * input_dim[1] * input_dim[2] / (2 ** (2 * self.num_conv - 2))
    hidden_dims = [last_convolution_size] + hidden_dims + [num_classes]

    for i in xrange(len(num_filters) + 1, len(num_filters) +  len(hidden_dims)):
        true_index = i - len(num_filters)
        self.params['W'+ str(i - 1)] = weight_scale * np.random.randn(hidden_dims[true_index - 1], hidden_dims[true_index])
        self.params['b'+ str(i - 1)] = np.zeros(hidden_dims[true_index])

    if self.use_batchnorm:
        conv_gammas = { 'gamma' + str(i): np.ones(num_filters[i]) for i in xrange(1, self.num_conv + 1)} 
        conv_betas = { 'beta' + str(i): np.zeros(num_filters[i]) for i in xrange(1, self.num_conv + 1)} 

        gammas = { 'gamma' + str(i-1): np.ones(hidden_dims[i - len(num_filters)]) for i in xrange(1+len(num_filters), len(num_filters) +  len(hidden_dims) - 1)} 
        betas = { 'beta' + str(i-1): np.zeros(hidden_dims[i - len(num_filters)]) for i in xrange(1+len(num_filters), len(num_filters) +  len(hidden_dims) - 1)} 

        self.params.update(conv_gammas)
        self.params.update(conv_betas)
        self.params.update(gammas)
        self.params.update(betas)

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):

    ## Forward 
    conv_out = None
    x_in = X

    conv_caches = []
    bn_caches1 = []
    relu_caches1 = []
    pool_caches = []
    dropout_caches1 = []

    mat_in = X
    for i in xrange(self.num_conv):
      conv_out, conv_cache = conv_forward_fast(mat_in, self.params['W' + str(i+1)], 
                                  self.params['b' + str(i+1)], self.conv_params[i])
      conv_caches.append(conv_cache)

      if self.use_batchnorm:
        conv_out, bn_cache = spatial_batchnorm_forward(conv_out, self.params['gamma' + str(i+1)],
                                  self.params['beta'+str(i+1)], self.bn_params[i])
        bn_caches1.append(bn_cache)

      conv_out, relu_cache = relu_forward(conv_out)

      relu_caches1.append(relu_cache)

      if i < self.num_conv-1:
        conv_out, pool_cache = max_pool_forward_fast(conv_out, self.pool_params[i])
        pool_caches.append(pool_cache)
      
      if self.use_dropout:
        conv_out, dropout_cache = dropout_forward(conv_out, self.dropout_param)
        dropout_caches1.append(dropout_cache)

      mat_in = conv_out

    mat_in = conv_out.reshape(X.shape[0], -1)
    scores = None

    affine_caches = []
    relu_caches2 = []
    bn_caches2 = []
    dropout_caches2 = []

    for i in xrange(self.num_fullconnected):
        p, affine_cache = affine_forward(mat_in, self.params['W'+str(i+1+self.num_conv)], self.params['b'+str(i+1+self.num_conv)])
        affine_caches.append(affine_cache)

        if i < self.num_fullconnected - 1:
            if self.use_batchnorm:
                gamma = self.params['gamma' + str(i + 1 + self.num_conv)]
                beta = self.params['beta' + str(i + 1 + self.num_conv)]
                
                p, bn_cache = batchnorm_forward(p, gamma, beta, self.bn_params[i + self.num_conv])
                bn_caches2.append(bn_cache)

            p, relu_cache = relu_forward(p)
            relu_caches2.append(relu_cache)
            
            if self.use_dropout:
                p, drop_cache = dropout_forward(p, self.dropout_param)
                dropout_caches2.append(drop_cache)

        mat_in = p
    scores = mat_in
    if y is None:
        return scores

    loss, grads = 0, {}

    loss, dLoss = softmax_loss(scores, y)

    sum_reg = 0
    for i in xrange(self.num_fullconnected):
        w = self.params['W'+str(i + 1 + self.num_conv)]
        b = self.params['b'+str(i + 1 + self.num_conv)]

        sum_reg += np.sum(w*w) + np.sum(b*b)

    loss += 0.5 * self.reg * sum_reg 
#
    din = dLoss
    for i in reversed(xrange(self.num_fullconnected)):

        p_index = i + 1 + self.num_conv
        if i < self.num_fullconnected - 1:
            if self.use_dropout:
                din = dropout_backward(din, dropout_caches2[i])

            din = relu_backward(din, relu_caches2[i]) 

            if self.use_batchnorm:
                din, dgamma, dbeta = batchnorm_backward_alt(din, bn_caches2[i])
                grads['gamma'+str(p_index)] = dgamma 
                grads['beta'+str(p_index)] = dbeta 

        x, w, b = affine_caches[i]
        dx, dw, db = affine_backward(din, affine_caches[i])
        
        grads['W'+str(p_index)] = dw + self.reg * w 
        grads['b'+str(p_index)] = db + self.reg * b 
        din = dx

    dFc = din.reshape(conv_out.shape)

    for i in reversed(xrange(self.num_conv)):

      if self.use_dropout:
        dFc = dropout_backward(dFc, dropout_caches1[i])

      if i < self.num_conv - 1:
        dFc = max_pool_backward_fast(dFc, pool_caches[i])


      dFc = relu_backward(dFc, relu_caches1[i])
      if self.use_batchnorm:
        dFc, dgamma, dbeta = spatial_batchnorm_backward(dFc, bn_caches1[i])
        grads['gamma'+str(i+1)] = dgamma 
        grads['beta'+str(i+1)] = dbeta 

      dFc, dw, db = conv_backward_fast(dFc, conv_caches[i])

      grads['W' + str(i+1)] = dw
      grads['b' + str(i+1)] = db

    return loss, grads

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
    self.params['W1'] = weight_scale * np.random.randn(num_filters, input_dim[0], filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)

    self.params['W2'] = weight_scale * np.random.randn(num_filters * input_dim[1] / 2 * input_dim[2] / 2, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)

    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
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
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv_out, conv_cache = conv_relu_pool_forward(X, self.params['W1'], self.params['b1'],
                            conv_param, pool_param)
    conv_out_flatten = conv_out.reshape(conv_out.shape[0], -1)
    hidden_out, hidden_cache = affine_relu_forward(conv_out_flatten, self.params['W2'], self.params['b2'])
    scores, last_cache = affine_forward(hidden_out, self.params['W3'], self.params['b3'])

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

    loss, dSoftmax = softmax_loss(scores, y)

    dLast, grads['W3'], grads['b3'] = affine_backward(dSoftmax, last_cache)
    dHidden, grads['W2'], grads['b2'] = affine_relu_backward(dLast, hidden_cache)
    dHidden_volume = dHidden.reshape(conv_out.shape)
    dX, grads['W1'], grads['b1'] = conv_relu_pool_backward(dHidden_volume, conv_cache)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
