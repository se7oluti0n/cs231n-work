import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    dScoredW = X[i]
    dCorrectdW = X[i]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      dMargindScorej = 1
      dMargindcorrect = -1
      if margin > 0:
        loss += margin
        dLdScorej = 1
        dLdCorrect = -1
        dW[:,j] += dScoredW * dLdScorej
        dW[:, y[i]] += dCorrectdW * dLdCorrect

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dLdR = 1
  dRdW = reg * W
  dW += dRdW * dLdR

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
    
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  
  #One -hot encoding the labels
  labels = (np.arange(W.shape[1]) == y[:,None]).astype(np.float32)
    
  correct_class_scores = np.sum(scores * labels, axis=1)
  margin = (scores - correct_class_scores[:, None] + 1) * (1 - labels)
  
  loss = np.sum(np.maximum(0, margin)) / X.shape[0]
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  dScoresdW = X.T
  
  dLdScore = (margin > 0).astype(np.float32)
  dLdCorrect = np.sum(dLdScore, axis=1)
  
  dW += dScoresdW.dot(dLdScore - dLdCorrect[:,None] * labels) 
  
  dW /= num_train

  dLdR = 1
  dRdW = reg * W
  dW += dRdW * dLdR
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
