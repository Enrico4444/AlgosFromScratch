import numpy as np

class GradientDescent():
    def __init__(self, X, y, w, loss, batch_size = None, reg_lambda = 0, update_X = False, seed = None):
        '''input:
        X: (n, m)
        y: (n, 1)
        loss: instace of class with at least two methods: "compute_loss" and "derivative"
        '''
        np.random.seed(seed)
        self.X = X
        self.y = y.reshape(-1,1)
        self.w = w
        self.loss = loss
        self.batch_size = batch_size if batch_size is not None else self.X.shape[0]
        self.reg_lambda = reg_lambda
        self.loss.compute_cost(self.X, self.w, self.y, self.reg_lambda)
        self.history = []
    
    def jacobian(self, x_batch, y_batch):
        self.gradient = self.loss.derivative(x_batch, self.w, y_batch, self.reg_lambda).reshape(x_batch.shape[1],1) # shape: mx1
    
    def weight_update(self, x_batch, y_batch, learning_rate):
        self.jacobian(x_batch, y_batch)
        self.w -= learning_rate * self.gradient
        
    def split_data(self):
        data = np.concatenate((self.X, self.y), axis=1)
        np.random.shuffle(data)
        self.X = data[:, :-1]
        self.y = data[:, -1].reshape(-1, 1)
        q = self.X.shape[0] // self.batch_size
        block_end = q * self.batch_size
        X_batches = np.split(self.X[:block_end], q) 
        y_batches = np.split(self.y[:block_end], q) 
        if self.X.shape[0] % self.batch_size != 0:
            X_batches = X_batches + [self.X[block_end:]] 
            y_batches = y_batches + [self.y[block_end:]]
        return X_batches, y_batches
        
    def gradient_descent(self, learning_rate = 0.1, max_iter = 100, threshold = 1e-3, debug = False):
        X_batches, y_batches = self.split_data()
        i = 0
        if debug:
            print('n batches {}\ninitial weights {}'.format(len(X_batches), self.w))
        while(i < max_iter and self.loss.cost > threshold):               
            for x_batch, y_batch in zip(X_batches, y_batches):
                self.weight_update(x_batch, y_batch, learning_rate)
            self.loss.compute_cost(self.X, self.w, self.y, self.reg_lambda)
            self.history.append(self.loss.cost)
            if debug:
                print('\n****iter {}'.format(i+1))
                print('gradient {}'.format(self.gradient))
                print('weights {}'.format(self.w))
                print('cost {}'.format(self.history[-1]))
            i += 1   