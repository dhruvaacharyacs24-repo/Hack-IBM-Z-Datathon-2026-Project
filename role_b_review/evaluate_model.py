
import numpy as np
import torch
import core
from metrics import report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Frozen model and threshold â€” do not tune on test results.
assert core.CRIT_T == 0.9
model = core.get_model()
X = np.load("X_test.npy")
y = np.load("y_test.npy")

assert X.shape == (39639, 1080)
assert y.shape == (39639,)
assert np.all(np.isfinite(X))
assert set(np.unique(y)).issubset({0, 1, 2})

# X_test is already preprocessed. Do not call core.predict().
model.eval()
all_probs = []

with torch.no_grad():
    for start in range(0, len(X), 512):
        batch = torch.from_numpy(X[start:start + 512])[:, None, :]
        probs = torch.softmax(model(batch), dim=1).cpu().numpy()
        all_probs.append(probs)

probs = np.concatenate(all_probs, axis=0)
assert probs.shape == (len(y), 3)
assert np.all(np.isfinite(probs))
assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

# Raw predictions: direct probability argmax.
raw_pred = probs.argmax(axis=1)

# Operational predictions: same Critical threshold as core.py.
operational_scores = probs.copy()
operational_scores[probs[:, 2] <= core.CRIT_T, 2] = 0.0
operational_pred = operational_scores.argmax(axis=1)

print("\n=== RAW ARGMAX TEST METRICS ===")
raw_results = report(y, raw_pred, probs)

print("\n=== OPERATIONAL TEST METRICS (CRIT_T=0.9) ===")
operational_results = report(y, operational_pred, probs)

# Save confusion matrix for operational predictions.
cm = confusion_matrix(y, operational_pred, labels=[0, 1, 2])
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=core.LEVELS
)
display.plot(values_format="d")
plt.title("V2 Operational Predictions (CRIT_T=0.9) - Untouched Test Set")
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.tight_layout()
plt.savefig("confusion.png", dpi=160)
plt.close()

print("\nSaved confusion.png")
print("Checkpoint:", core.CKPT)
print("Critical threshold:", core.CRIT_T)
print("Test windows evaluated:", len(y))
print("Raw argmax confusion matrix:\n",
      raw_results["confusion_matrix"])
print("Operational confusion matrix:\n", cm)

