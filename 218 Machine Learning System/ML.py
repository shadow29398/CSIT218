import pandas as pd
import numpy as np
import time
import os  # added by Alex R
import joblib  # Added by Alex R
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
try:
    from scipy.stats import ttest_ind
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False



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
        return model, accuracy, f1, elapsed      




def get_prediction_scores(model, X):
    """
    Returns a 1D array of 'prediction scores' for the positive class.
    - If model supports predict_proba: use proba[:, 1]
    - Else if model supports decision_function: use that (scaled score)
    - Else: returns None
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # If decision_function is 2D, take positive column
        if scores.ndim > 1 and scores.shape[1] == 2:
            return scores[:, 1]
        return scores
    return None


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

    results = []

    for name, model in algorithms:
        fitted_model, accuracy, f1, elapsed = pl.train_and_evaluate(model, name)
        print(f"{name:<25} {accuracy:<15.4f} {f1:<15.4f} {elapsed:<10.6f}")

        # Compute prediction scores (probabilities or decision scores)
        y_pred  = fitted_model.predict(pl.X_test)
        y_score = get_prediction_scores(fitted_model, pl.X_test)

        # Confusion matrix
        cm = confusion_matrix(pl.y_test, y_pred, labels=[0, 1])
        print(f"\nConfusion Matrix — {name}\n{cm}\n")

        # Classification report
        print(f"Classification report — {name}")
        print(classification_report(pl.y_test, y_pred, digits=4))

        # Plot & save confusion matrix
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest')
        ax.set_title(f'Confusion Matrix — {name}')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha='center', va='center')
        plt.tight_layout()
        fig.savefig(f"cm_{name.replace(' ', '_')}.png", dpi=180)
        plt.close(fig)

        # ROC-AUC if we have scores. The closer to 0.5 the AUC value is the worse the model is at predicting the the classes, the classes being "High Quality Wine" and "Low Quality Wine".
        auc = None
        if y_score is not None:
            try:
                auc = roc_auc_score(pl.y_test, y_score)
                print(f"ROC-AUC — {name}: {auc:.4f}")
            except Exception:
                pass

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Time (s)": elapsed,
            "ROC-AUC": auc if auc is not None else np.nan
        })

        # Save per-sample predictions with scores
        out_pred = pd.DataFrame({
            "y_true": pl.y_test,
            "y_pred": y_pred,
            "score": y_score if y_score is not None else np.nan
        })
        out_pred.to_csv(f"predictions_{name.replace(' ', '_')}.csv", index=False)

    # Convert results list to a DataFrame
    results_df = pd.DataFrame(results)

    # Plot the results
    ax = results_df.plot(
        x="Model",
        y=["Accuracy", "F1 Score"],
        kind="bar"
    )
    ax.set_title("Model Comparison on Wine Quality (High = quality ≥ 6)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=0)
    plt.tight_layout()

    # Save chart and display
    plt.savefig("model_comparison.png", dpi=200)
    print("\n[INFO] Saved performance chart as model_comparison.png")
    plt.show()

    # Format table for display
    table_df = results_df.copy()

    # Format numeric columns with fixed decimal places
    for col in ("Accuracy", "F1 Score", "ROC-AUC"):
        if col in table_df.columns:
            table_df[col] = table_df[col].map(
                lambda x: f"{x:.4f}" if pd.notnull(x) else "NA"
            )
    table_df["Time (s)"] = table_df["Time (s)"].map(lambda x: f"{x:.2f}")

    print("\n=== Model Comparison Table ===")
    print(table_df.to_string(index=False))


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

    # t-scores per feature: high (target=1) vs low (target=0)
    if SCIPY_OK:
        print("\nTop features by |t-score| (Welch’s t-test high vs low):")
        feats = [c for c in df.columns if c not in ("quality", "target")]
        ts = {}
        high_rows = df[df["target"] == 1][feats]
        low_rows  = df[df["target"] == 0][feats]
        for col in feats:
            tstat, pval = ttest_ind(high_rows[col], low_rows[col], equal_var=False, nan_policy='omit')
            ts[col] = (abs(tstat), tstat, pval)
        # sort by absolute t-statistic
        top = sorted(ts.items(), key=lambda kv: kv[1][0], reverse=True)[:5]
        for col, (abs_t, tstat, p) in top:
            print(f"{col:<22} t={tstat:+.3f}  |t|={abs_t:.3f}  p={p:.2e}")
    else:
        print("\n[WARN] SciPy not found — skipping t-scores. Install with: pip install scipy")


if __name__ == "__main__":
    main()
