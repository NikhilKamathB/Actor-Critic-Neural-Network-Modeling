import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import Color, str2bool


class LinearClassificationBinaryData:

    def __init__(self, m:float = 1.0, c:float = 3.0, sample_size:int = 1000, write_output:bool = True, verbose:bool = True) -> None:
        '''
            Initialize the linear data generator definitions.
            Input parameters:
                m: slope of the line
                c: intercept of the line
                sample_size: number of samples to generate
                write_output: whether to write the generated data to a csv file
                verbose: whether to print the generated data
        '''
        self.m = m
        self.c = c
        self.sample_size = sample_size
        self.write_output = write_output
        self.verbose = verbose
    
    def generate(self, output_file:str = "linear_data") -> None:
        '''
            Generate the linear data and write it to a csv file.
            Input parameters:
                output_file: name of the output file
        '''
        x1 = np.random.uniform(low=-1.0, high=1.0, size=(self.sample_size, 1))
        x2 = self.m * x1 + self.c + np.random.normal(loc=0.0, scale=1.0, size=(self.sample_size, 1))
        x = np.concatenate((x1, x2), axis=1)
        y = np.where(self.m * x1 + self.c - x2 > 0, 1, 0)
        if self.verbose: self.visualize(x=x, y=y)
        if self.write_output:
            pd.DataFrame(x).to_csv(output_file + ".csv", index=False, header=False)
            pd.DataFrame(y).to_csv(output_file + "_labels.csv", index=False, header=False)
    
    def visualize(self, x:np.ndarray, y:np.ndarray) -> None:
        '''
            Visualize the generated data.
            Input parameters:
                x: input data
                y: output labels
        '''
        _ = plt.figure(figsize=(7, 7))
        plt.scatter(x[:, 0], x[:, 1], c=y)
        # plot best fit boundary
        x_sample = np.linspace(-1, 1, 5)
        y_sample = eval("self.m * x_sample + self.c")
        plt.plot(x_sample, y_sample, 'black')
        plt.show()


def main(args):
    if args.linear:
        linear_data = LinearClassificationBinaryData(m=args.line_slope, c=args.line_intercept, sample_size=args.sample_size, write_output=args.write_output, verbose=args.verbose)
        linear_data.generate(output_file=args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Data.')
    parser.add_argument('-l', '--linear', type=str2bool, default='y', metavar="\b", help="Generate linear data.")
    parser.add_argument('-lm', '--line-slope', type=float, default=1.0, metavar="\b", help="Slope of the line.")
    parser.add_argument('-lc', '--line-intercept', type=float, default=3.0, metavar="\b", help="Intercept of the line.")
    parser.add_argument('-s', '--sample-size', type=int, default=1000, metavar="\b", help="Number of samples to generate.")
    parser.add_argument('-o', '--output-file', type=str, default="synth_data", metavar="\b", help="Name of the output file.")
    parser.add_argument('-w', '--write-output', type=str2bool, default='y', metavar="\b", help="Write to a output file.")
    parser.add_argument('-v', '--verbose', type=str2bool, default='y', metavar="\b", help="Display the generated data.")
    main(args=parser.parse_args())