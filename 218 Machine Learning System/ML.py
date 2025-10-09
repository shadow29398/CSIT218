import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MLPipeline:
    def __init__(self, data_path):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, path):
        pass
        
    def preprocess(self):
        pass
        
    def train_model(self, algorithm):
        pass
        
    def evaluate(self):
        pass


def main():

    pipeline = MLPipeline('')
    pipeline.load_data()
    pipeline.preprocess()
    
    algorithms = [

    ]
    
    print(f"{'Algorithm':<25} {'Accuracy':<15} {'F1-Score':<15} {'Time':<10}")
    print("-"*70)
    
    for name, model in algorithms:
        accuracy, f1, time = pipeline.train_and_evaluate(model)
        print(f"{name:<25} {accuracy:<15.4f} {f1:<15.4f} {time:<10.6f}")