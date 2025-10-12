import pandas as pd
import numpy as np
import time
import os  # added by Alex R
import joblib  # Added by Alex R
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class MLPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()

    def load_data(self):              
        # The Wine Quality dataset has a target column 'quality'.
        self.data = pd.read_csv(self.data_path, sep=';')  # dataset uses ';'
        print("[INFO] Data loaded successfully.")
        print("[INFO] Shape:", self.data.shape)
        print("[INFO] Columns:", list(self.data.columns))

    def preprocess(self):
        # Splits into train and test sets
        # columns except 'quality'
        X = self.data.drop('quality', axis=1)
        y = self.data['quality']
        y = (y >= 6).astype(int)

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Standardize
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print("[INFO] Preprocessing completed.")
        print("[INFO] Training set:", self.X_train.shape)
        print("[INFO] Test set:", self.X_test.shape)

    def train_and_evaluate(self, model, name):
        # trains given model, evaluates accuracy and F1-score.
        # Added by Alex R: Try to load saved model first and trains if not found, saves model after training
        model_filename = f"{name.replace(' ', '_')}_model.pkl"
        if os.path.exists(model_filename):
            print(f"[INFO] Found saved model for {name}. Loading instead of retraining...")
            start = time.time()
            model = joblib.load(model_filename)
            end = time.time()
            elapsed = end - start
        else:
            print(f"[INFO] Training {name}...")
            start = time.time()
            model.fit(self.X_train, self.y_train)
            end = time.time()
            joblib.dump(model, model_filename)
            elapsed = end - start
            print(f"[INFO] Model saved as {model_filename} ({elapsed:.2f}s)")

        # Evaluate
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        return accuracy, f1, elapsed        



        """" original code of function pre modification by Alex R. use this if you want the model to be trained every time
    def train_and_evaluate(self, model, name):
        # trains given model, evaluates accuracy and F1-score.
        start = time.time()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        end = time.time()

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        return accuracy, f1, end - start
        """


def main():
    pl = MLPipeline('winequality-red.csv')    #feel free to change directory

    # Load n preprocess data
    pl.load_data()
    pl.preprocess()

    # Choose models to train
    algorithms = [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("SVM", SVC())
    ]

    # Output da results
    print(f"{'Algorithm':<25} {'Accuracy':<15} {'F1-Score':<15} {'Time':<10}")
    print("-" * 70)

    for name, model in algorithms:
        accuracy, f1, elapsed = pl.train_and_evaluate(model)
        print(f"{name:<25} {accuracy:<15.4f} {f1:<15.4f} {elapsed:<10.6f}")

    # Added by Alex R
    # ------------------------------------------------------------
    # Compares chemical traits of high vs low quality wines.
    # 'target' = 1 if quality >= 6 (high), else 0 (low).
    # You can tweak the 6 threshold to change what counts as "high".
    # Calculates mean feature values for both groups and shows which
    # features differ the most between them.
    # ------------------------------------------------------------
    df = pl.data.copy()
    df['target'] = (df['quality'] >= 6).astype(int)   # tweak threshold if you guys want
    high = df[df['target'] == 1].drop(columns=['quality','target']).mean()
    low  = df[df['target'] == 0].drop(columns=['quality','target']).mean()
    diff = (high - low).sort_values(key=np.abs, ascending=False)
    # Added by Alex R

    print("\nTop mean differences (High - Low quality):")
    for feat, val in diff.head(5).items():
        print(f"{feat:<22} {val:+.4f}")


if __name__ == "__main__":
    main()
