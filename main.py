from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

digits = load_digits()

X = digits.data
y = digits.target

print("Dataset:", X.shape)
print("Labels:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

class MyKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)

        if not 1 <= self.k <= len(self.X_train):
            raise ValueError("k must be between 1 and the training size")

    def predict_one(self, sample):
        differences = self.X_train - sample
        distances = np.sqrt(np.sum(differences ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = self.y_train[nearest_indices]
        counts = np.bincount(nearest_labels, minlength=10)
        prediction = np.argmax(counts)

        return prediction

    def predict(self, X_test):
        predictions = []

        for sample in X_test:
            prediction = self.predict_one(sample)
            predictions.append(prediction)

        return np.array(predictions)

my_model = MyKNN(k=3)
my_model.fit(X_train, y_train)
my_predictions = my_model.predict(X_test)

sklearn_model = KNeighborsClassifier(n_neighbors=3)
sklearn_model.fit(X_train, y_train)
sklearn_predictions = sklearn_model.predict(X_test)

my_accuracy = accuracy_score(y_test, my_predictions)
sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

print("\n=== Model Comparison ===")
print(f"My KNN Accuracy:       {my_accuracy:.2%}")
print(f"Scikit-learn Accuracy: {sklearn_accuracy:.2%}")

agreement = np.mean(my_predictions == sklearn_predictions)
print(f"Prediction Agreement:  {agreement:.2%}")

wrong_indices = np.where(y_test != my_predictions)[0]
print(f"Wrong Predictions:     {len(wrong_indices)}")

results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(exist_ok=True)

cm = confusion_matrix(y_test, my_predictions, labels=np.arange(10))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=np.arange(10)
)

fig, ax = plt.subplots(figsize=(8, 7))
display.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("KNN From Scratch - Confusion Matrix")
fig.tight_layout()
fig.savefig(results_dir / "confusion_matrix.png", dpi=160)
plt.close(fig)

if len(wrong_indices) > 0:
    n = min(10, len(wrong_indices))
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3))

    axes = np.atleast_1d(axes)

    for ax, i in zip(axes, wrong_indices[:n]):
        image = X_test[i].reshape(8, 8)

        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"True: {y_test[i]}\nPred: {my_predictions[i]}"
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(results_dir / "wrong_predictions.png", dpi=160)
    plt.close(fig)

print("\nResults saved in:", results_dir)