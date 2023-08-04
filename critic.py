import sys
import time
from src.nn import *
from src.pipeline import MLClassificationPipeline


class Critic:

    def __init__(self, train_data_file_name:str, train_label_file_name:str, test_data_file_name:str, test_label_file_name:str) -> None:
        self.epochs = 100000
        self.step_size = 100000
        self.lr = 1e-3
        self.train_data_file_name = train_data_file_name
        self.train_label_file_name = train_label_file_name
        self.test_data_file_name = test_data_file_name
        self.test_label_file_name = test_label_file_name

    def run(self) -> None:
        _start_time = time.time()
        ml_pipeline = MLClassificationPipeline(
            epochs=self.epochs,
            step_size=self.step_size,
            lr=self.lr,
            train_data_file_name=self.train_data_file_name, 
            train_label_file_name=self.train_label_file_name, 
            test_data_file_name=self.test_data_file_name, 
            test_label_file_name=self.test_label_file_name
        )
        ml_pipeline.prepare_data()
        ml_pipeline.build(
            architecture={
                "L_1": LinearLayer(in_features=ml_pipeline.num_features, out_features=32, weight_initialization_method="xaiver"),
                "L_1_RELU": Tanh(),
                "L_2": LinearLayer(in_features=32, out_features=32, weight_initialization_method="xaiver"),
                "L_2_RELU": Tanh(),
                "L_3": LinearLayer(in_features=32, out_features=16, weight_initialization_method="xaiver"),
                "L_3_CostFunction": Tanh(),
                "L_5": LinearLayer(in_features=16, out_features=1, weight_initialization_method="xaiver"),
                "L_5_SIGMOID": Sigmoid(),
                "L_5_BCE": BCE() 
            }
        )
        ml_pipeline.train()
        ml_pipeline.test()
        print(f"Total time take for executions in minutes = {(time.time()-_start_time)/60} minutes.")


if __name__ == "__main__":
    train_data_file_name, train_label_file_name, test_data_file_name, test_label_file_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) == 5 else None
    critic = Critic(train_data_file_name=train_data_file_name, train_label_file_name=train_label_file_name, test_data_file_name=test_data_file_name, test_label_file_name=test_label_file_name)
    critic.run()