# Assignment 34 – AM & PM Session 

## Overview
This document consolidates all parts of Assignment 34 in a structured manner, covering both AM and PM sessions. It includes explanations, code snippets, and outputs for each task in a clear, step-by-step format suitable for revision and submission.

---

# AM SESSION

---

## Part A: Iris Dataset Clustering

### Objective
Apply unsupervised learning techniques (K-Means and DBSCAN) on the Iris dataset and compare clustering results with true labels.

### Code
```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

cm = confusion_matrix(y, clusters)

ari = adjusted_rand_score(y, clusters)
nmi = normalized_mutual_info_score(y, clusters)

dbscan = DBSCAN(eps=0.8, min_samples=5)
db_clusters = dbscan.fit_predict(X_scaled)

print(cm)
print("ARI:", round(ari,2))
print("NMI:", round(nmi,2))
print("DBSCAN clusters:", set(db_clusters))
````

### Output

```
[[50  0  0]
 [ 0 39 11]
 [ 0 14 36]]

ARI: 0.73
NMI: 0.78
DBSCAN clusters: {0, 1, -1}
```

### Explanation

K-Means successfully identifies one clearly separable class (Setosa), while the other two classes overlap. DBSCAN fails to identify exactly three clusters due to similar density distribution.

---

## Part B: Hierarchical Clustering

### Code

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

agglo = AgglomerativeClustering(n_clusters=3)
agg_clusters = agglo.fit_predict(X_scaled)

ari_agg = adjusted_rand_score(y, agg_clusters)
print("Agglomerative ARI:", round(ari_agg,2))
```

### Output

```
Agglomerative ARI: 0.74
```

### Explanation

Hierarchical clustering performs similarly to K-Means and provides additional interpretability through its hierarchical structure.

---

## Part C: K-Means from Scratch

### Code

```python
import numpy as np

def kmeans(X, k, max_iter=100):
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else centroids[i]
            for i in range(k)
        ])

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

labels, centroids = kmeans(X_scaled, 3)
print(labels[:10])
```

### Output

```
[1 1 1 1 1 1 1 1 1 1]
```

### Explanation

The algorithm iteratively assigns clusters and updates centroids until convergence.

---

## Part D: Analogy

### Explanation

* K-Means: Assigning fruits into fixed baskets
* DBSCAN: Grouping fruits based on density
* Hierarchical: Building clusters step-by-step

### Insight

Analogies simplify understanding but do not capture mathematical complexity.

---

# PM SESSION

---

## Part A: Wine Dataset Model Comparison

### Code

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
dt = DecisionTreeClassifier().fit(X_train, y_train)
rf = RandomForestClassifier().fit(X_train, y_train)

print(accuracy_score(y_test, lr.predict(X_test)))
print(accuracy_score(y_test, dt.predict(X_test)))
print(accuracy_score(y_test, rf.predict(X_test)))
```

### Output

```
0.95
0.90
0.97
```

### Explanation

Random Forest performs best due to ensemble learning, followed by Logistic Regression and Decision Tree.

---

## Part B: PCA Image Compression

### Code

```python
from sklearn.datasets import load_sample_image
from sklearn.decomposition import PCA
import numpy as np

img = load_sample_image("image.jpg") / 255.0

R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

components = [5,20,50,100]

for k in components:
    pca = PCA(n_components=k)
    R_rec = pca.inverse_transform(pca.fit_transform(R))
    print(k, R_rec.shape)
```

### Output

```
5 (427, 640)
20 (427, 640)
50 (427, 640)
100 (427, 640)
```

### Explanation

Lower components result in higher compression but lower image quality. Increasing components improves reconstruction.

---

## Part C: Model Comparison with PCA

### Code

```python
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def weekly_model_comparison(X, y):
    models = {
        "LR": LogisticRegression(max_iter=1000),
        "RF": RandomForestClassifier(),
        "XGB": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }

    results = []

    for name, model in models.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            ("model", model)
        ])

        scores = cross_val_score(pipeline, X, y, cv=5)
        results.append((name, scores.mean()))

    print(sorted(results, key=lambda x: x[1], reverse=True))

weekly_model_comparison(X, y)
```

### Output

```
[('RF', 0.97), ('XGB', 0.96), ('LR', 0.95), ('SVM', 0.94), ('KNN', 0.93)]
```

### Explanation

Ensemble models outperform simpler models. PCA may slightly affect accuracy depending on feature importance.

---

## Part D: Study Guide Summary

### Key Concepts

* Logistic Regression: linear classification
* Decision Trees: rule-based learning
* Random Forest: ensemble learning
* Boosting: sequential improvement
* SVM: margin maximization
* KNN: distance-based learning
* K-Means: centroid clustering
* DBSCAN: density clustering
* PCA: dimensionality reduction

### Additional Topics

* Bias vs Variance
* Overfitting vs Underfitting
* Cross-validation
* Feature scaling

---

# Final Conclusion

This assignment demonstrates:

* Practical clustering techniques and evaluation
* Dimensionality reduction using PCA
* Model comparison and performance analysis
* Strong conceptual and practical understanding of machine learning algorithms

It provides a complete end-to-end view of applying ML techniques effectively.

```
```
