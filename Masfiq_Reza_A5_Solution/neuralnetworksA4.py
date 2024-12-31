# A4 Solution
# CS545 Fall 2024

import numpy as np
import optimizers as opt
import sys  # for sys.float_info.epsilon

######################################################################
## class NeuralNetwork()
######################################################################

class NeuralNetwork():

    """
    A class that represents a neural network for nonlinear regression.

    Attributes
    ----------
    n_inputs : int
        The number of values in each sample
    n_hidden_units_by_layers : list of ints, or empty
        The number of units in each hidden layer.
        Its length specifies the number of hidden layers.
    n_outputs : int
        The number of units in output layer
    all_weights : one-dimensional numpy array
        Contains all weights of the network as a vector
    Ws : list of two-dimensional numpy arrays
        Contains matrices of weights in each layer,
        as views into all_weights
    all_gradients : one-dimensional numpy array
        Contains all gradients of mean square error with
        respect to each weight in the network as a vector
    Grads : list of two-dimensional numpy arrays
        Contains matrices of gradients weights in each layer,
        as views into all_gradients
    total_epochs : int
        Total number of epochs trained so far
    performance_trace : list of floats
        Mean square error (unstandardized) after each epoch
    n_epochs : int
        Number of epochs trained so far
    X_means : one-dimensional numpy array
        Means of the components, or features, across samples
    X_stds : one-dimensional numpy array
        Standard deviations of the components, or features, across samples
    T_means : one-dimensional numpy array
        Means of the components of the targets, across samples
    T_stds : one-dimensional numpy array
        Standard deviations of the components of the targets, across samples
    debug : boolean
        If True, print information to help with debugging
        
    Methods
    -------
    train(Xtrain, Ttrain, Xvalidate, Tvalidate, n_epochs, batch_size=-1, method='sgd', learning_rate=None, verbose=True)
        Trains the network using input and target samples in rows of Xtrain and Ttrain.
        Sets final weight values to ones for which error is lowest on Xvalidate and Tvalidate

    use(X)
        Applies network to inputs X and returns network's output

    get_performance_trace()
        Returns list of performance values (MSE or -Log Likelihood) from each epoch.
    
    set_debug(v)
        Print debugging information if v is True
    
    _make_weights_and_views(shapes)
        Creates all initial weights and views for each layer. Also used for gradients and views for each layer.

    """

    def __init__(self, n_inputs, n_hiddens_each_layer, n_outputs):
        """Creates a neural network with the given structure

        Parameters
        ----------
        n_inputs : int
            The number of values in each sample
        n_hidden_units_by_layers : list of ints, or empty
            The number of units in each hidden layer.
            Its length specifies the number of hidden layers.
        n_outputs : int
            The number of units in output layer

        Returns
        -------
        NeuralNetwork object
        """

        self.n_inputs = n_inputs
        self.n_hiddens_each_layer = n_hiddens_each_layer
        self.n_outputs = n_outputs

        # Create one-dimensional numpy array of all weights with random initial values

        shapes = []
        n_in = n_inputs
        for nu in self.n_hiddens_each_layer + [n_outputs]:
            shapes.append((n_in + 1, nu))
            n_in = nu

        # Build list of views (pairs of number of rows and number of columns)
        # by reshaping corresponding elements from vector of all weights 
        # into correct shape for each layer.        

        self.all_weights, self.Ws = self._make_weights_and_views(shapes)
        self.all_gradients, self.Grads = self._make_weights_and_views(shapes)

        self.X_means = None
        self.X_stds = None
        self.T_means = None
        self.T_stds = None

        self.total_epochs = 0
        self.performance = None
        self.performance_trace = []
        self.best_epoch = np.inf
        self.best_validation_error = np.inf

        self.debug = False
        
    def __repr__(self):
        return f'NeuralNetwork({self.n_inputs}, {self.n_hiddens_each_layer}, {self.n_outputs})'

    def __str__(self):
        if self.total_epochs > 0:
            s = f'{self.__repr__()} trained for {self.total_epochs} epochs'
            s += f'\n  with final errors of {self.performance_trace[-1][0]:.4f} train {self.performance_trace[-1][1]:.4f} validation.'
            s += f'\n  Network weights set to best weights from epoch {self.best_epoch} for validation error of {self.best_validation_error}.'
            return s
        else:
            return f'{self.__repr__()} has not been trained.'
 
    def _make_weights_and_views(self, shapes):
        """Creates vector of all weights and views for each layer

        Parameters
        ----------
        shapes : list of pairs of ints
            Each pair is number of rows and columns of weights in each layer.
            Number of rows is number of inputs to layer (including constant 1).
            Number of columns is number of units, or outputs, in layer.

        Returns
        -------
        Vector of all weights, and list of views into this vector for each layer
        """

        n_weights = np.sum([np.prod(Wshape) for Wshape in shapes])
        all_weights = np.random.uniform(-1, 1, n_weights)

        # Build weight matrices as list of views (pairs of number of rows and number 
        # of columns) by reshaping corresponding elements from vector of all weights 
        # into correct shape for each layer.  
        # Do the same to make list of views for gradients.
 
        Ws = []
        first_index = 0
        for Wshape in shapes:
            last_index = first_index + np.prod(Wshape)
            nin = Wshape[0]
            W = all_weights[first_index:last_index].reshape(Wshape)
            W /= np.sqrt(nin)
            Ws.append(W)
            first_index = last_index

        # Set output layer weights to zero.
        Ws[-1][:] = 0
        
        return all_weights, Ws

    def set_debug(self, d):
        """Set or unset printing of debugging information.

        Parameters
        ----------
        d : boolean
            If True, print debugging information. 
        """
        
        self.debug = d
        if self.debug:
            print('Debugging information will now be printed.')
        else:
            print('No debugging information will be printed.')
        
    def train(self, Xtrain, Ttrain, Xvalidate, Tvalidate, n_epochs, batch_size=-1,
              method='sgd', learning_rate=0.1, momentum=0.9, verbose=True):
        """Updates the weights.

        Parameters
        ----------
        Xtrain : two-dimensional numpy array 
            number of training samples  by  number of input components
        Ttrain : two-dimensional numpy array
            number of training samples  by  number of output components
        Xvalidate : two-dimensional numpy array 
            number of validation samples  by  number of input components
        Tvalidate : two-dimensional numpy array
            number of validationg samples  by  number of output components
        n_epochs : int
            Number of passes to take through all samples
        batch_size : int
            Size of mini-batches. If -1, then just one batch of all training data
        method : str
            'sgd', 'adamw', or 'scg'
        learning_rate : float
            Controls the step size of each update, only for sgd and adamw
        momentum : float
            Controls amount of previous weight update to add to current weight update, only for sgd
        verbose: boolean
            If True, progress is shown with print statements

        Returns
        -------
        self : NeuralNetwork instance
        """

        # Calculate and assign standardization parameters

        if self.X_means is None:
            self.X_means = Xtrain.mean(axis=0)
            self.X_stds = Xtrain.std(axis=0)
            self.X_stds[self.X_stds == 0] = 1
            self.T_means = Ttrain.mean(axis=0)
            self.T_stds = Ttrain.std(axis=0)

        # Standardize X's and T's.  Assign back to Xtrain, Train, Xvalidate, Tvalidate.

        Xtrain = (Xtrain - self.X_means) / self.X_stds
        Ttrain = (Ttrain - self.T_means) / self.T_stds
        Xvalidate = (Xvalidate - self.X_means) / self.X_stds
        Tvalidate = (Tvalidate - self.T_means) / self.T_stds
        
        # Instantiate Optimizers object by giving it vector of all weights
        
        optimizer = opt.Optimizers(self.all_weights)

        # Define function to convert mean-square error to root-mean-square error,
        # Here we use a lambda function just to illustrate its use.  
        # We could have also defined this function with
        # def error_convert_f(err):
        #     return np.sqrt(err)

        error_convert_f = lambda err: np.sqrt(err)
        error_convert_name = 'RMSE'
        
        # Call the requested optimizer method to train the weights.

        if method == 'sgd':
            optimizer_method = optimizer.sgd
        elif method == 'adamw':
            optimizer_method = optimizer.adamw
        elif method == 'scg':
            optimizer_method = optimizer.scg
        else:
            raise Exception("method must be 'sgd', 'adamw', or 'scg'")
            
        performance_trace = optimizer_method(Xtrain, Ttrain, Xvalidate, Tvalidate,
                                             self._error_f, self._gradient_f,
                                             n_epochs=n_epochs, batch_size=batch_size,
                                             learning_rate=learning_rate,
                                             momentum=momentum,
                                             error_convert_f=error_convert_f,
                                             error_convert_name=error_convert_name,
                                             verbose=verbose)

        self.total_epochs += n_epochs
        self.performance_trace += performance_trace

        self.best_epoch = optimizer.best_epoch
        self.best_validation_error = optimizer.best_val_error

        # Return neural network object to allow applying other methods
        # after training, such as:    Y = nnet.train(X, T, 100, 0.01).use(X)

        return self

    def _add_ones(self, X):
        return np.insert(X, 0, 1, 1)
    
    def _forward(self, X):
        """Calculate outputs of each layer given inputs in X.
        
        Parameters
        ----------
        X : input samples, standardized with first column of constant 1's.

        Returns
        -------
        Standardized outputs of all layers as list, include X as first element.
        """

        self.Zs = [X]

        # Append output of each layer to list in self.Zs, then return it.

        for W in self.Ws[:-1]:  # forward through all but last layer
            self.Zs.append(np.tanh(self._add_ones(self.Zs[-1]) @ W))
        last_W = self.Ws[-1]
        self.Zs.append(self._add_ones(self.Zs[-1]) @ last_W)

        return self.Zs

    # Function to be minimized by optimizer method, mean squared error
    def _error_f(self, X, T):
        """Calculate output of net given input X and its mean squared error.
        Function to be minimized by optimizer.

        Parameters
        ----------
        X : two-dimensional numpy array, standardized
            number of samples  by  number of input components
        T : two-dimensional numpy array, standardized
            number of samples  by  number of output components

        Returns
        -------
        Standardized mean square error as scalar float that is the mean
        square error over all samples and all network outputs.
        """

        if self.debug:
            print('in _error_f: X[0] is {} and T[0] is {}'.format(X[0], T[0]))
        Zs = self._forward(X)
        mean_sq_error = np.mean((T - Zs[-1]) ** 2)
        if self.debug:
            print(f'in _error_f: mse is {mean_sq_error}')
        return mean_sq_error

    # Gradient of function to be minimized for use by optimizer method
    def _gradient_f(self, X, T):
        """Returns gradient wrt all weights. Assumes _forward already called
        so input and all layer outputs stored in self.Zs

        Parameters
        ----------
        X : two-dimensional numpy array, standardized
            number of samples  x  number of input components
        T : two-dimensional numpy array, standardized
            number of samples  x  number of output components

        Returns
        -------
        Vector of gradients of mean square error wrt all weights
        """

        # Assumes _forward just called with layer outputs saved in self.Zs.
        n_samples = X.shape[0]
        n_outputs = T.shape[1]
        n_layers = len(self.n_hiddens_each_layer) + 1

        # delta is delta matrix to be back propagated.
        # Dividing by n_samples and n_outputs is due the calculation of
        # the gradient of the mean squared error.

        delta = - 2 * (T - self.Zs[-1]) / (n_samples * n_outputs)

        # Step backwards through the layers to back-propagate the error (delta)
        # Use member function _backpropagate so that it can be used again in
        #  class NeuralNetworkClassifier and any other derived classes.
        
        self._backpropagate(delta)

        return self.all_gradients

    def _backpropagate(self, delta):
        """Backpropagate output layer delta through all previous layers,
        setting self.Grads, the gradient of the objective function wrt weights in each layer.

        Parameters
        ----------
        delta : two-dimensional numpy array of output layer delta values
            number of samples  x  number of output components
        """

        n_layers = len(self.n_hiddens_each_layer) + 1
        if self.debug:
            print('in _backpropagate: first delta calculated is\n{}'.format(delta))
            
        # for G, W, Z in zip(reversed(self.Ws), reversed(self.Grads), reversed(self.Zs[:-1])):
        #     print(f'{G.shape=} {W.shape=} {Z.shape=}')
            
        for W, G, Z in zip(reversed(self.Ws), reversed(self.Grads), reversed(self.Zs[:-1])):
            G[:] = self._add_ones(Z).T @ delta
            delta = delta @ W[1:, :].T * (1 - Z ** 2)
            if self.debug:
                print('in _backpropagate: next delta is\n{}'.format(delta))


    def use(self, X):
        """Return the output of the network for input samples as rows in X.
        X assumed to not be standardized.

        Parameters
        ----------
        X : two-dimensional numpy array
            number of samples  by  number of input components, unstandardized

        Returns
        -------
        Output of neural network, unstandardized, as numpy array
        of shape  number of samples  by  number of outputs
        """

        # Standardize X
        X = (X - self.X_means) / self.X_stds
        Zs = self._forward(X)
        # Unstandardize output Y before returning it
        return Zs[-1] * self.T_stds + self.T_means

    def get_performance_trace(self):
        """Returns list of unstandardized root-mean square error for each epoch"""
        return self.performance_trace


######################################################################
## class NeuralNetworkClassifier(NeuralNetwork)
######################################################################

class NeuralNetworkClassifier(NeuralNetwork):

    def __repr__(self):
        return f'NeuralNetworkClassifier({self.n_inputs}, {self.n_hiddens_each_layer}, {self.n_outputs})'

    def __str__(self):
        if self.total_epochs > 0:
            s = f'{self.__repr__()} trained for {self.total_epochs} epochs'
            s += f'\n  with final likelihoods of {self.performance_trace[-1][0]:.4f} train {self.performance_trace[-1][1]:.4f} validation.'
            s += f'\n  Network weights set to best weights from epoch {self.best_epoch} for validation likelihood of {self.best_validation_error:.4f}.'
            return s
        else:
            return f'{self.__repr__()} has not been trained.'


    def _make_indicator_vars(self, T):
        """Convert column matrix of class labels (ints or strs) into indicator variables

        Parameters
        ----------
        T : two-dimensional array of all ints or all strings
            number of samples by 1
        
        Returns
        -------
        Two dimensional array of indicator variables. Each row is all 0's except one value of 1.
            number of samples by number of output components (number of classes)
        """

        # Make sure T is two-dimensional. Should be n_samples x 1.
        if T.ndim == 1:
            T = T.reshape((-1, 1))    
        return (T == np.unique(T)).astype(float)  # to work on GPU

    def _softmax(self, Y):
        """Convert output Y to exp(Y) / (sum of exp(Y)'s)

        Parameters
        ----------
        Y : two-dimensional array of network output values
            number of samples by number of output components (number of classes)

        Returns
        -------
        Two-dimensional array of indicator variables representing Y
            number of samples by number of output components (number of classes)
        """

        # Trick to avoid overflow
        maxY = Y.max() 
        expY = np.exp(Y - maxY)
        denom = expY.sum(1).reshape((-1, 1))
        Y_softmax = expY / (denom + sys.float_info.epsilon)
        return Y_softmax

    # Function to be minimized by optimizer method, mean squared error
    def _neg_log_likelihood_f(self, X, T):
        """Calculate output of net given input X and the resulting negative log likelihood.
        Function to be minimized by optimizer.

        Parameters
        ----------
        X : two-dimensional numpy array, standardized
            number of samples  by  number of input components
        T : two-dimensional numpy array of class indicator variables
            number of samples  by  number of output components (number of classes)

        Returns
        -------
        Negative log likelihood as scalar float.
        """
        
        if self.debug:
            print('In _neg_log_likelihood_f: arguments are')
            print('X (standardized):')
            print(X)
            print('T (indicator variables):')
            print(T)
            
        Zs = self._forward(X)

        if self.debug:
            print('Initial Zs is:')
            print(Zs)
            
        Y_softmax = self._softmax(Zs[-1])

        if self.debug:
            print('Result of _softmax is:')
            print(Y_softmax)

        if self.debug:
            print('Result of np.log(Y + sys.float_info.epsilon) is:')
            print(np.log(Y_softmax + sys.float_info.epsilon))

        neg_mean_log_likelihood =  - np.mean(T * np.log(Y_softmax + sys.float_info.epsilon))
        
        if self.debug:
            print('_neg_log_likelihood_f returns:')
            print(neg_mean_log_likelihood)
            
        return neg_mean_log_likelihood

    def _gradient_f(self, X, T):
        """Returns gradient wrt all weights. Assumes _forward (from NeuralNetwork class)
        has already called so input and all layer outputs stored in self.Zs

        Parameters
        ----------
        X : two-dimensional numpy array, standardized
            number of samples  x  number of input components
        T : two-dimensional numpy array of class indicator variables
            number of samples  by  number of output components (number of classes)

        Returns
        -------
        Vector of gradients of negative log likelihood wrt all weights
        """

        # ... code lines to calculate first delta (for output layer)

        n_samples = X.shape[0]
        n_outputs = T.shape[1]

        Y = self.Zs[-1]
        Y_softmax = self._softmax(Y)
        delta = - (T - Y_softmax) / (n_samples * n_outputs)

        
        # Step backwards through the layers to back-propagate the error (delta)
        self._backpropagate(delta)
        
        # print('new gradients:\n',self.all_gradients)
        return self.all_gradients

    def train(self, Xtrain, Ttrain, Xvalidate, Tvalidate,
              n_epochs, batch_size=-1, method='sgd',
              learning_rate=0.1, momentum=0.9, verbose=True):

        """Updates the weights.

        Parameters
        ----------
        Xtrain : two-dimensional numpy array 
            number of training samples  by  number of input components
        Ttrain : two-dimensional numpy array
            number of training samples  by  number of output components
        Xvalidate : two-dimensional numpy array 
            number of validation samples  by  number of input components
        Tvalidate : two-dimensional numpy array
            number of validationg samples  by  number of output components
        n_epochs : int
            Number of passes to take through all samples
        batch_size : int
            Size of mini-batches. If -1, then just one batch of all training data
        method : str
            'sgd', 'adamw', or 'scg'
        learning_rate : float
            Controls the step size of each update, only for sgd and adamw
        momentum : float
            Controls amount of previous weight update to add to current weight update, only for sgd
        verbose: boolean
            If True, progress is shown with print statements

        Returns
        -------
        self : NeuralNetworkClassifier instance
        """

        # Calculate and assign standardization parameters

        if self.X_means is None:
            self.X_means = Xtrain.mean(axis=0)
            self.X_stds = Xtrain.std(axis=0)
            self.X_stds[self.X_stds == 0] = 1

        # Standardize X's, but not T's.  Assign back to Xtrain and Xvalidate.

        Xtrain = (Xtrain - self.X_means) / self.X_stds
        Xvalidate = (Xvalidate - self.X_means) / self.X_stds

        # Assign class labels, based on Ttrain, to self.classes
        # Create indicator values representation from target labels in T.

        self.classes = np.unique(Ttrain)
        Ttrain_ind_vars = self._make_indicator_vars(Ttrain)
        Tvalidate_ind_vars = self._make_indicator_vars(Tvalidate)

        # Define function to convert negative log likelihood values to likelihood values.
        error_convert_f = lambda nll: np.exp(-nll)
        error_convert_name = 'Likelihood'
        
        # Instantiate Optimizers object by giving it vector of all weights.
        optimizer = opt.Optimizers(self.all_weights)

        # Call the requested optimizer method to train the weights.

        if method == 'sgd':
            optimizer_method = optimizer.sgd
        elif method == 'adamw':
            optimizer_method = optimizer.adamw
        elif method == 'scg':
            optimizer_method = optimizer.scg
        else:
            raise Exception("method must be 'sgd', 'adamw', or 'scg'")

        performance_trace = optimizer_method(Xtrain, Ttrain_ind_vars, Xvalidate, Tvalidate_ind_vars,
                                       self._neg_log_likelihood_f, self._gradient_f,
                                       n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate,
                                       error_convert_f=error_convert_f,
                                       error_convert_name=error_convert_name,
                                       momentum=momentum, verbose=verbose)

        self.total_epochs += len(performance_trace)
        self.performance_trace += performance_trace
        self.best_epoch = min(self.best_epoch, optimizer.best_epoch)
        self.best_validation_error = min(self.best_validation_error, optimizer.best_val_error)

        return self

    def use(self, X):
        """Return the predicted class and probabilities for input samples as rows in X.
        X assumed to not be standardized.

        Parameters
        ----------
        X : two-dimensional numpy array, unstandardized input samples by rows
            number of samples  by  number of input components, unstandardized

        Returns
        -------
        Predicted classes : two-dimensional array of predicted classes for each sample
            number of samples by 1  of ints or strings, depending on how target classes were specified
        Class probabilities : two_dimensional array of probabilities of each class for each sample
            number of samples by number of outputs (number of classes)
        """

        # Standardize X
        X = (X - self.X_means) / self.X_stds

        # Calculate softmax of outputs
        Y_softmax = self._softmax(self._forward(X)[-1])
        probs = Y_softmax
        
        # Assign classes for each input sample, based on argmax of softmax values
        classes = self.classes[np.argmax(probs, axis=1)].reshape(-1, 1)

        # Return classes (of each sample) and the probabilities (softmax values)

        return classes, probs

