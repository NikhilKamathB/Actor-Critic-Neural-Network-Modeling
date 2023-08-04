import math
import random
import numpy as np
from typing import Generator


class DataGenerator:

    def __init__(self, batch_size:int, X:np.ndarray, Y:np.ndarray = None, is_stochastic:bool = False) -> None:
        self.batch_size =  batch_size
        self.is_stochastic = is_stochastic
        if Y is not None:
            self.mode = "train" 
            self.data = np.concatenate((X, Y), axis=1)
        else:
            self.mode = "test" 
            self.data = X
        self.data = self.data[np.random.permutation(len(self.data))]
    
    def __len__(self) -> int:
        return int(math.ceil(len(self.data)/self.batch_size)) if not self.is_stochastic else 1
    
    def process_result(self, data:np.ndarray) -> tuple:
        if self.mode == "train":
            return (data[:, :-1], np.expand_dims(data[:, -1], axis=1))
        return (data, None)
    
    def get_item(self) -> Generator[tuple, None, None]:
        if self.is_stochastic:
            while True:
                idx = random.randint(0, len(self.data)-1)
                X, Y = self.process_result(data=np.expand_dims(self.data[idx], axis=0)) 
                yield (X, Y)
        else:
            ptr = 0
            while True:
                if ptr * self.batch_size >= len(self.data):
                    ptr = 0
                    self.data = self.data[np.random.permutation(len(self.data))]
                    continue
                else:
                    X, Y = self.process_result(data=self.data[ptr*self.batch_size: (ptr+1)*self.batch_size, :])
                    ptr += 1
                    yield (X, Y)