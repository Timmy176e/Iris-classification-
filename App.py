import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

st.set_page_config(page_title="Iris KNN Classifier", page_icon="🌸", layout="wide")
st.title("🌸 Iris Species Classifier — K-Nearest Neighbors")
st.caption("DecodeLabs AI Engineering Internship — Project 2: Data Classification Using AI")

iris = load_iris()
X_full = pd.DataFrame(iris.data, columns=iris.feature_names)
y_full = pd.Series(iris.target)

st.sidebar.header("Settings")
k_value = st.sidebar.slider("K (number of neighbors)", 1, 20, 5)
test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, step=0.05)
show_imbalance_demo = st.sidebar.checkbox("Show imbalance demo (shrink virginica class)")

tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Elbow Method", "Model & Evaluation", "Imbalance Demo"])

# Shared prep, used by tabs 2 and 3
X_s, y_s = shuffle(X_full, y_full, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_s, y_s, test_size=test_size, random_state=42)
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)

with tab1:
    st.subheader("Dataset Overview")
    df_display = X_full.copy()
    df_display["species"] = y_full.map(dict(enumerate(iris.target_names)))
    st.write(f"Shape: {df_display.shape}")
    st.dataframe(df_display.head(10))
    st.write("Class balance:")
    st.bar_chart(df_display["species"].value_counts())
    st.write("Feature summary:")
    st.dataframe(df_display.describe())

with tab2:
    st.subheader("Finding Optimal K — Elbow Method")
    k_range = range(1, 21)
    error_rates = []
    for k in k_range:
        m = KNeighborsClassifier(n_neighbors=k)
        m.fit(X_tr_scaled, y_tr)
        preds = m.predict(X_te_scaled)
        error_rates.append((preds != y_te).mean())

    fig, ax = plt.subplots()
    ax.plot(list(k_range), error_rates, marker="o")
    ax.axvline(k_value, color="red", linestyle="--", label=f"Selected K={k_value}")
    ax.set_xlabel("K value")
    ax.set_ylabel("Error rate")
    ax.set_title("Elbow Method for Optimal K")
    ax.legend()
    st.pyplot(fig)

with tab3:
    st.subheader(f"Model trained with K={k_value}")
    model = KNeighborsClassifier(n_neighbors=k_value)
    model.fit(X_tr_scaled, y_tr)
    predictions = model.predict(X_te_scaled)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Confusion Matrix")
        cm = confusion_matrix(y_te, predictions)
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names).plot(ax=ax_cm, cmap="Blues", colorbar=False)
        st.pyplot(fig_cm)
    with col2:
        st.write("Classification Report")
        report = classification_report(y_te, predictions, target_names=iris.target_names, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Try your own flower")
    sl = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8)
    sw = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
    pl = st.slider("Petal length (cm)", 1.0, 7.0, 4.3)
    pw = st.slider("Petal width (cm)", 0.1, 2.5, 1.3)
    user_point = scaler.transform([[sl, sw, pl, pw]])
    user_pred = model.predict(user_point)[0]
    st.success(f"Predicted species: **{iris.target_names[user_pred]}**")

with tab4:
    if show_imbalance_demo:
        st.subheader("Why Accuracy Can Lie — Imbalance Demo")
        rng = np.random.RandomState(42)
        virginica_idx = y_full[y_full == 2].index
        keep_virginica = rng.choice(virginica_idx, size=5, replace=False)
        other_idx = y_full[y_full != 2].index
        keep_idx = np.concatenate([other_idx, keep_virginica])
        X_imb, y_imb = X_full.loc[keep_idx], y_full.loc[keep_idx]

        st.write("New class balance:", y_imb.value_counts().to_dict())

        X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(
            X_imb, y_imb, test_size=0.2, random_state=42, stratify=y_imb
        )
        scaler_i = StandardScaler()
        X_tr_i_scaled = scaler_i.fit_transform(X_tr_i)
        X_te_i_scaled = scaler_i.transform(X_te_i)

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_tr_i_scaled, y_tr_i)
        dummy_preds = dummy.predict(X_te_i_scaled)

        knn_i = KNeighborsClassifier(n_neighbors=5)
        knn_i.fit(X_tr_i_scaled, y_tr_i)
        knn_i_preds = knn_i.predict(X_te_i_scaled)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Lazy (majority-class) Classifier**")
            st.text(classification_report(y_te_i, dummy_preds, target_names=iris.target_names, zero_division=0))
        with col2:
            st.write("**KNN Classifier**")
            st.text(classification_report(y_te_i, knn_i_preds, target_names=iris.target_names, zero_division=0))

        st.info("Both models look okay on accuracy, but recall on the rare 'virginica' class tells the real story.")
    else:
        st.write("Check the box in the sidebar to see the imbalance demo.")
