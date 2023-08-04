import numpy as np
from src.data import DataGenerator
from src.nn import NNSequentialModule


class MLClassificationPipeline:

    def __init__(
            self,
            train_data_file_name:str,
            train_label_file_name:str,
            test_data_file_name:str,
            test_label_file_name:str,
            need_val_set:bool = False,
            batch_size:int = 32, 
            lr:float = 0.03, 
            epochs:int = 10,
            step_size:int = 10,
            is_stochastic:bool = False,
            verbose:bool = True
        ) -> None:
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.step_size = step_size
        self.is_stochastic = is_stochastic
        self.train_data_file_name = train_data_file_name
        self.train_label_file_name = train_label_file_name
        self.test_data_file_name = test_data_file_name
        self.test_label_file_name = test_label_file_name
        self.need_val_set = need_val_set
        self.verbose = verbose
    
    def read_csv(self, file_name:str) -> np.ndarray:
        return np.loadtxt(fname=file_name, delimiter=',')
    
    def get_num_classes(self, Y:np.ndarray) -> int:
        return int(Y.max() + 1)
    
    def set_training_data(self, train_data_file_name:str, train_label_file_name:str) -> None:
        self.X_train = self.read_csv(file_name=train_data_file_name)
        self.Y_train = np.expand_dims(self.read_csv(file_name=train_label_file_name), axis=1)
        self.num_features = self.X_train.shape[-1]
        self.num_classes = self.get_num_classes(Y=self.Y_train)
    
    def set_testing_data(self, test_data_file_name:str, test_label_file_name:str) -> None:
        self.X_test = self.read_csv(file_name=test_data_file_name)
        self.Y_test = np.expand_dims(self.read_csv(file_name=test_label_file_name), axis=1) if test_label_file_name else None

    def write_csv(self, output:np.ndarray, test_label_file_name:str = "test_predictions.csv") -> None:
        np.savetxt(test_label_file_name, output.astype(int), delimiter=',', fmt='%d')

    def build(self, architecture:dict) -> None:
        if self.verbose: print("Building model ...")
        self.model = NNSequentialModule(
            batch_size=self.batch_size, 
            num_features=self.num_features, 
            lr=self.lr, 
            architecture=architecture, 
            reg_lambda=None
        )

    def one_hot_encode(self, c:int, Y:np.ndarray) -> np.ndarray:
        return np.squeeze(np.eye(N=c)[Y.astype(int)])

    def prepare_data(self) -> None:
        if self.verbose: print("Preparing training data ...")
        self.set_training_data(train_data_file_name=self.train_data_file_name, train_label_file_name=self.train_label_file_name,)
        if self.verbose: print("Preparing testing data ...")
        self.set_testing_data(test_data_file_name=self.test_data_file_name, test_label_file_name=self.test_label_file_name)
    
    def get_train_data_generator(self, X:np.ndarray, Y:np.ndarray) -> object:
        return DataGenerator(batch_size=self.batch_size, X=X, Y=Y, is_stochastic=self.is_stochastic)
    
    def report(self, loss:float, confusion_matrix:np.ndarray):
        true_positive, false_positive, false_negative, true_negative = confusion_matrix.flatten()
        accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
        recall = true_positive / max((true_positive + false_negative), 1e-9)
        precision = true_positive / max((true_positive + false_positive), 1e-9)
        f1_score = 2 * ((precision * recall) / max((precision + recall), 1e-9))
        return (loss, accuracy, recall, precision, f1_score)
    
    def compute_confusion_matrix(self, Y_hat:np.ndarray, Y:np.ndarray, verbose:bool=False) -> np.ndarray:
        Y_hat = Y_hat.round()
        false_positive = np.sum(Y_hat > Y)
        false_negative = np.sum(Y > Y_hat)
        true_positive = np.sum((Y_hat==1) & (Y==1))
        true_negative = np.sum((Y_hat==0) & (Y==0))
        confusion_matrix = np.array([[true_positive, false_positive], [false_negative, true_negative]])
        if verbose: print("Confusion matrix:\n", confusion_matrix)
        return confusion_matrix

    def train(self) -> None:
        if self.verbose: print("Begining to train the model ...")
        train_data_generator = self.get_train_data_generator(X=self.X_train, Y=self.Y_train)
        train_loss, train_accuracy = [], []
        for epoch in range(self.epochs):
            train_batch_running_loss, train_batch_running_accuracy = 0, 0
            for _ in range(len(train_data_generator)):
                X_train_batch, Y_train_batch = next(train_data_generator.get_item())
                Y_hat, loss = self.model.fit(X=X_train_batch, Y=Y_train_batch)
                loss, accuracy, _, _, _ = self.report(loss=loss, confusion_matrix=self.compute_confusion_matrix(Y_hat=Y_hat, Y=Y_train_batch))
                train_batch_running_loss += loss
                train_batch_running_accuracy += accuracy
            if self.verbose and epoch % self.step_size == 0:
                epoch_train_loss = train_batch_running_loss/len(train_data_generator)
                epoch_train_accuracy = train_batch_running_accuracy/len(train_data_generator)
                train_loss.append(epoch_train_loss)
                train_accuracy.append(epoch_train_accuracy)
                print(f"   Epoch {epoch}:  Train Loss = {epoch_train_loss}  Train Accuracy = {epoch_train_accuracy}") 

    def test(self) -> None:
        if self.Y_test is None: return self.predict()
        if self.verbose: print("Begining to test the model ...")
        Y_hat, loss = self.model.test(X=self.X_test, Y=self.Y_test)
        if self.verbose: print("Testing done ...")
        self.predict()
    
    def predict(self) -> None:
        if self.verbose: print("Writing output ...")
        self.write_csv(output=self.model.predict(X=self.X_test))
        if self.verbose: print("Writing output done ...")