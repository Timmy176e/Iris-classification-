# 🌸 Iris Species Classifier — KNN

A K-Nearest Neighbors (KNN) classifier that predicts Iris flower species (setosa, versicolor, virginica) from four measurements: sepal length, sepal width, petal length, and petal width.

Built as **Project 2 — Data Classification Using AI** for the DecodeLabs AI Engineering Industrial Training Program (Batch 2026).

🔗 **Live app:** _add your Streamlit Community Cloud link here_

## What it does

- Loads and explores the classic Iris dataset (150 samples, 3 balanced classes)
- Shuffles and splits the data into training/testing sets
- Scales features with `StandardScaler` so no single measurement dominates the distance calculation KNN relies on
- Uses the **elbow method** to help choose K (number of neighbors)
- Trains a KNN classifier and evaluates it with a confusion matrix, precision, recall, and F1-score — not just raw accuracy
- Includes an interactive demo showing why **accuracy alone can be misleading on imbalanced data** (a model can score high accuracy while completely missing a rare class)
- Lets you enter your own flower measurements and get a live prediction

## Tech stack

- Python
- scikit-learn
- pandas / numpy
- Streamlit (web interface)
- Matplotlib (plots)

## Running locally

```bash
git clone <your-repo-url>
cd iris-knn-app
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
iris-knn-app/
├── app.py            # Streamlit app
├── requirements.txt  # Dependencies
└── README.md
```

## Key concepts demonstrated

- Supervised learning / classification
- Feature scaling and why it matters for distance-based algorithms
- Train/test splitting and data leakage avoidance
- Hyperparameter tuning (K) via the elbow method
- Evaluation beyond accuracy: confusion matrix, precision, recall, F1

---
Built collaboratively — DecodeLabs Batch 2026
