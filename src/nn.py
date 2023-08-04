import numpy as np


################################################################################################################################################################################################################################################################
# Model layers go here.
################################################################################################################################################################################################################################################################

class LinearLayer:

    def __init__(self, in_features:int, out_features:int, reg_lambda:float = 1e-3, weight_initialization_method:str = "he") -> None:
        self.in_features, self.out_features = in_features, out_features
        self.reg_lambda = reg_lambda
        self.params, self.gradients = dict(), dict()
        self.weight_initialization_method = weight_initialization_method.lower()
        self.initialize_params(method=self.weight_initialization_method)
    
    def initialize_params(self, method:str = "he") -> None:
        if method == "xavier": # generally used with sigmoid and tanh activations
            self.params['W'] = np.random.randn(self.in_features, self.out_features) * (np.sqrt(1/self.in_features))
        if method == "he": # generally used with relu activations
            self.params['W'] = np.random.randn(self.in_features, self.out_features) * (np.sqrt(2/self.in_features))
        else:
            self.params['W'] = np.random.normal(loc=0, scale=0.1, size=(self.in_features, self.out_features))
        self.params['b'] = np.random.normal(loc=0, scale=0.1, size=(1, self.out_features))
        self.gradients['W'] = np.zeros((self.in_features, self.out_features))
        self.gradients['b'] = np.zeros((1, self.out_features))
    
    def forward(self, X:np.ndarray) -> np.ndarray:
        return (X @ self.params['W']) + self.params['b']
    
    def backward(self, X:np.ndarray, gradient:np.ndarray) -> np.ndarray:
        self.gradients['W'] = X.T @ gradient
        if self.reg_lambda: self.gradients['W'] += (self.reg_lambda / self.params['W'].shape[0]) * self.params['W']
        self.gradients['b'] = np.sum(gradient, axis=0)
        return gradient @ self.params['W'].T
    
    def __str__(self) -> str:
        return f"LinearLayer(in_features={self.in_features}, out_features={self.out_features}, reg_lambda={self.reg_lambda}, weight_init='{self.weight_initialization_method}')"


################################################################################################################################################################################################################################################################
# Model activations go here.
################################################################################################################################################################################################################################################################

class Relu:

    def forward(self, X:np.ndarray) -> np.ndarray:
        self.mask = (X > 0) * 1
        return np.maximum(X, 0)

    def backward(self, X:np.ndarray, gradient:np.ndarray) -> np.ndarray:
        return gradient * self.mask
    
    def __str__(self) -> str:
        return f"Relu()"


class Sigmoid:

    def sigmoid(self, X:np.ndarray) -> np.ndarray:
        return 1.0 / (1 + np.exp(-1 * X))

    def forward(self, X:np.ndarray) -> np.ndarray:
        return self.sigmoid(X=X)

    def backward(self, X:np.ndarray, gradient:np.ndarray) -> np.ndarray:
        s = self.sigmoid(X=X)
        s_dash = s * (1 - s)
        return gradient * s_dash
    
    def __str__(self) -> str:
        return f"Sigmoid()"


class Tanh:

    def tanh(self, X:np.ndarray) -> np.ndarray:
        return np.tanh(X)

    def forward(self, X:np.ndarray) -> np.ndarray:
        return self.tanh(X=X)

    def backward(self, X:np.ndarray, gradient:np.ndarray) -> np.ndarray:
        t = self.tanh(X=X)
        t_dash = 1 - np.square(t)
        return gradient * t_dash
    
    def __str__(self) -> str:
        return f"Tanh()"
    

################################################################################################################################################################################################################################################################
# Model loss goes here.
################################################################################################################################################################################################################################################################

class BCE:

    def forward(self, Y_hat:np.ndarray, Y:np.ndarray, eps:np.float32 = 1e-13) -> float:
        return -np.mean((Y * np.log(np.maximum(Y_hat, eps)) + ((1-Y) * np.log(np.maximum(1-Y_hat, eps)))))
    
    def backward(self, Y_hat:np.ndarray, Y:np.ndarray, eps:np.float32 = 1e-13) -> np.ndarray:
        return -(Y / np.maximum(Y_hat, eps)) + ((1-Y) / (np.maximum(1-Y_hat, eps)))
    
    def __str__(self) -> str:
        return f"BCE()"
    

################################################################################################################################################################################################################################################################
# Model regularization goes here.
################################################################################################################################################################################################################################################################

class L2Regularization:

    def __init__(self, reg_lambda:float = 1e-3) -> None:
        self.reg_lambda = reg_lambda
    
    def regularize(self, model:object, batch_size:int = 32):
        reg_loss = 0
        for _, layer in model.items():
            if hasattr(layer, "params"):
                reg_loss += np.sum(np.square(layer.params['W']))
        return self.reg_lambda * reg_loss * 0.5 * (1 / batch_size)


################################################################################################################################################################################################################################################################
# Model building goes here.
################################################################################################################################################################################################################################################################

class NNSequentialModule:

    def __init__(self, batch_size:int, num_features:int, lr:float, architecture:dict, reg_lambda:float = 3e-3, verbose:bool = True) -> None:
        self.model_state = {}
        self.batch_size = batch_size
        self.num_features = num_features
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.verbose = verbose
        self.build_model(architecture=architecture)
    
    def build_model(self, architecture:dict) -> None:
        self.model = {}
        for k, v in architecture.items():
            self.model[k] = v
        if self.verbose: self.model_summary()
        
    def model_summary(self, hash_multiplier:int = 100, dash_multiplier:int = 25) -> str:
        msg = ""
        msg += '#' * hash_multiplier
        msg += '\n'
        msg += '-' * dash_multiplier
        msg += "Model summary"
        msg += '-' * dash_multiplier
        msg += '\n'
        for k, v in self.model.items():
            msg += f"{k} = {v.__str__()}"
            msg += '\n'
        msg += '#' * hash_multiplier
        return msg
    
    def forward(self, X:np.ndarray, Y:np.ndarray) -> tuple:
        model_keys = tuple(self.model.keys())
        self.model_state[f"{model_keys[0]}_FORWARD"] = self.model[model_keys[0]].forward(X=X)
        for itr, k in enumerate(model_keys[1: -1]):
            self.model_state[f"{k}_FORWARD"] = self.model[k].forward(X=self.model_state[f"{model_keys[itr]}_FORWARD"])
        loss = self.model[model_keys[-1]].forward(Y_hat=self.model_state[f"{model_keys[-2]}_FORWARD"], Y=Y)
        if self.reg_lambda: loss += L2Regularization(reg_lambda=self.reg_lambda).regularize(model=self.model, batch_size=self.batch_size)
        self.model_state[f"{model_keys[-1]}_FORWARD"] = self.model_state[f"{model_keys[-2]}_FORWARD"]
        self.model_state[f"LOSS"] = loss
        return (self.model_state[f"{model_keys[-2]}_FORWARD"], loss)

    def backward(self, X:np.ndarray, Y_hat:np.ndarray, Y:np.ndarray) -> None:
        model_keys = tuple(self.model.keys())[::-1]
        self.model_state[f"{model_keys[0]}_BACKWARD"] = self.model[model_keys[0]].backward(Y_hat=Y_hat, Y=Y)
        for itr, k in enumerate(model_keys[1: -1]):
            self.model_state[f"{k}_BACKWARD"] = self.model[k].backward(X=self.model_state[f"{model_keys[itr+2]}_FORWARD"], gradient=self.model_state[f"{model_keys[itr]}_BACKWARD"])
        self.model_state[f"{model_keys[-1]}_BACKWARD"] = self.model[model_keys[-1]].backward(X=X, gradient=self.model_state[f"{model_keys[-2]}_BACKWARD"])

    def step(self) -> None:
        for _, layer in self.model.items():
            if hasattr(layer, "params"):
                for k, _ in layer.params.items():
                    gradients = layer.gradients[k]
                    layer.params[k] -= self.lr * gradients

    def fit(self, X:np.ndarray, Y:np.ndarray):
        Y_hat, loss = self.forward(X=X, Y=Y)
        self.backward(X=X, Y_hat=Y_hat, Y=Y)
        self.step()
        return Y_hat, loss
    
    def test(self, X:np.ndarray, Y:np.ndarray):
        Y_hat, loss = self.forward(X=X, Y=Y)
        return Y_hat, loss
    
    def predict(self, X:np.ndarray):
        pred_model_state = {}
        model_keys = tuple(self.model.keys())
        pred_model_state[f"{model_keys[0]}_FORWARD"] = self.model[model_keys[0]].forward(X=X)
        for itr, k in enumerate(model_keys[1: -1]):
            pred_model_state[f"{k}_FORWARD"] = self.model[k].forward(X=pred_model_state[f"{model_keys[itr]}_FORWARD"])
        return pred_model_state[f"{model_keys[-2]}_FORWARD"].round()

    def __str__(self) -> str:
        return f"NNSequentialModule():\n{self.model_summary()}"