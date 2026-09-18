"""Small NumPy MLP: explicit ReLU/linear, Adam, MSE/Huber, best validation state.

No heavyweight GPU framework is needed for ~248 features and small trial sets.
L2 penalty is weight_decay/2 * sum(W**2), excluding biases (coupled Adam L2).
"""
import copy
import numpy as np
from sklearn.preprocessing import StandardScaler


class MLPRegressor:
    def __init__(self, config, seed=42):
        self.config, self.seed = copy.deepcopy(config), seed
        self.scaler = StandardScaler()

    def _forward(self, x):
        activations = [x]
        for w, b in zip(self.weights_, self.biases_):
            x = x@w + b
            if len(activations) < len(self.weights_):
                x = np.maximum(x, 0)
            activations.append(x)
        return activations

    def _loss_grad(self, pred, y):
        error = pred - y
        if self.config['loss'] == 'mse':
            return float(np.mean(error**2)), 2*error/error.size
        if self.config['loss'] == 'huber':
            delta = self.config['huber_delta']
            loss = np.where(np.abs(error) <= delta, .5*error**2, delta*(np.abs(error)-.5*delta))
            return float(np.mean(loss)), np.clip(error, -delta, delta)/error.size
        raise ValueError('Loss must be mse or huber')

    def fit(self, X, y, X_val, y_val, progress=None, should_stop=None):
        c = self.config
        if not c['hidden'] or min(c['hidden']) < 1 or c['batch_size'] < 1 or c['max_epochs'] < 1:
            raise ValueError('Invalid MLP size/epoch parameters')
        if c['learning_rate'] <= 0 or c['weight_decay'] < 0 or c['huber_delta'] <= 0 or c['patience'] < 1:
            raise ValueError('Invalid optimizer or stopping parameters')
        rng = np.random.default_rng(self.seed)
        x = self.scaler.fit_transform(X)
        xv = self.scaler.transform(X_val)
        dims = [x.shape[1], *c['hidden'], y.shape[1]]
        self.weights_ = [rng.normal(0, np.sqrt(2/a), (a, b)) for a, b in zip(dims[:-1], dims[1:])]
        self.biases_ = [np.zeros(b) for b in dims[1:]]
        parameters = self.weights_ + self.biases_
        m, v = [[np.zeros_like(p) for p in parameters] for _ in range(2)]
        self.history_ = []
        best, stale, step, state = np.inf, 0, 0, None
        for epoch in range(1, c['max_epochs']+1):
            if should_stop and should_stop():
                raise InterruptedError('Training cancelled')
            order = rng.permutation(len(x))
            for start in range(0, len(x), c['batch_size']):
                idx = order[start:start+c['batch_size']]
                activations = self._forward(x[idx])
                _, delta = self._loss_grad(activations[-1], y[idx])
                gw, gb = [None]*len(self.weights_), [None]*len(self.weights_)
                for layer in range(len(self.weights_)-1, -1, -1):
                    gw[layer] = activations[layer].T@delta + c['weight_decay']*self.weights_[layer]
                    gb[layer] = delta.sum(axis=0)
                    if layer:
                        delta = (delta@self.weights_[layer].T)*(activations[layer] > 0)
                step += 1
                for j, (p, grad) in enumerate(zip(parameters, gw+gb)):
                    m[j] = .9*m[j] + .1*grad
                    v[j] = .999*v[j] + .001*grad**2
                    p -= c['learning_rate']*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
            train_loss = self._loss_grad(self._forward(x)[-1], y)[0]
            val_loss = self._loss_grad(self._forward(xv)[-1], y_val)[0]
            val_mse = float(np.mean((self._forward(xv)[-1]-y_val)**2))
            if not np.isfinite(val_loss):
                raise ValueError('MLP diverged; reduce learning rate/check data')
            row = {'epoch': epoch, 'train_loss': train_loss, 'validation_loss': val_loss,
                   'validation_mse': val_mse}
            self.history_.append(row)
            if progress:
                progress({'stage': 'train', 'model': 'mlp', **row})
            # Validation MSE is the common selection criterion for all models.
            if val_mse < best:
                improvement = best - val_mse
                best, state = val_mse, copy.deepcopy((self.weights_, self.biases_))
                self.best_epoch_ = epoch
                stale = 0 if improvement > c['min_delta'] else stale+1
            else:
                stale += 1
            if stale >= c['patience']:
                break
        if state is None:
            raise ValueError('No MLP state trained')
        self.weights_, self.biases_ = state
        return self

    def predict(self, X):
        return self._forward(self.scaler.transform(X))[-1]
