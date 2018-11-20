import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    correct_class_ind = y[i]
    f = X[i].dot(W)
    stabilty_factor = -np.max(f)
    f = f + stabilty_factor
    nominator = np.exp(f[correct_class_ind])
    denominator = np.sum(np.exp(f))
    loss += -np.log(nominator / denominator)
    #Gradient calc
    p = lambda k: np.exp(f[k])/denominator
    for j in xrange(num_classes):
      pj = p(j)
      dW[:, j] += (pj - (j == y[i]))*X[i]

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  dW /= num_train
  dW += reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = X.dot(W)
  stabilty_factor = -np.max(f, axis=1).reshape((num_train, -1))
  f = np.exp(f + stabilty_factor)
  f_correct_class = f[range(num_train), y].reshape((num_train, -1))
  f_sum = np.sum(f, axis=1).reshape((num_train, -1))
  p = f_correct_class/f_sum
  loss = np.sum(-np.log(p))
  loss /= num_train
  # Gradient calc
  f_all = f/f_sum
  indicator = np.zeros(f_all.shape)
  indicator[range(num_train), y] = 1
  dW = X.T.dot(f_all - indicator)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

