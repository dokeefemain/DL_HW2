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
  dW = np.zeros_like(W).transpose()

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. So use stable version   #
  # of softmax. Don't forget the regularization!                              #
  #############################################################################
  num_train = X.shape[0]
  reg1 = 0
  summ = 0
  scores = X[0].dot(W)
  count = 0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    exp = np.exp(scores)
    prob = exp / np.sum(exp)
    for j in range(10):
      if j != y[i]:
        dW[j,:] += X[i] * prob[j]
      else:
        dW[j,:] += X[i] * (prob[j] - 1)

    Li = -1 * np.log(prob)
    loss += np.sum(Li) / len(Li) + np.sum(W ** 2) * reg

  loss = loss / num_train
  dW = dW / num_train
  dW = np.array(dW).transpose()

  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)
  exp = np.exp(scores)
  prob = np.multiply(exp, 1 / np.sum(exp, axis=1).reshape((-1, 1)))

  X1 = np.tile(X,10)

  prob1 =np.repeat(prob.reshape(-1,1),X.shape[1]).reshape(X1.shape)
  tmp = np.multiply(X1,prob1)


  prob_y = (np.tile(np.choose(y, np.array(prob).transpose()).reshape((-1,1)),X.shape[1])) - 1
  tmp_y = np.multiply(X, prob_y)

  y_range = (y) * X.shape[1]

  y_range = y_range.reshape((-1,1)) + np.arange(X.shape[1])
  x_ind = np.repeat(np.arange(np.array(y_range).shape[0]),y_range.shape[1]).reshape((-1,1))

  tmp[np.array(x_ind).flatten(),np.array(y_range).flatten()] = np.array(tmp_y).flatten()

  sumW = np.sum(tmp,axis = 0)
  sumW = sumW.reshape((dW.shape[1],dW.shape[0]))
  dW = np.array(sumW).transpose()
  dW = dW / num_train
  dW += reg * W

  Li = -1 * np.log(prob)
  loss_r = np.sum(Li, axis=1) / len(Li[0]) + np.sum(W ** 2) * reg
  loss = np.sum(loss_r) / num_train


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW