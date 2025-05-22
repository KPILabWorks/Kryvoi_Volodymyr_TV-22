import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.60, random_state=42)

np.random.seed(42)
outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
X = np.vstack((X, outliers))

model = IsolationForest(contamination=0.06, random_state=42)
model.fit(X)

labels = model.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[labels == 1][:, 0], X[labels == 1][:, 1], c='blue', label='Нормальні дані')
plt.scatter(X[labels == -1][:, 0], X[labels == -1][:, 1], c='red', label='Аномалії')
plt.title("Виявлення аномалій за допомогою Isolation Forest")
plt.xlabel("Ознака 1")
plt.ylabel("Ознака 2")
plt.legend()
plt.grid(True)
plt.show()
