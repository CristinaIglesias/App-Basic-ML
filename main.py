import streamlit as st
from streamlit_lottie import st_lottie
import numpy as np
import requests
import json

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_lottiefile(filepath: str):
    with open(filepath, "r")as f:
        return json.load(f)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_coding = load_lottiefile("machine_learning.json")
lottie_Machine_learning = load_lottieurl( "https://assets6.lottiefiles.com/packages/lf20_jtvduiqm.json")

st_lottie (
    lottie_coding,
    speed=1,
    reverse=False,
    loop=True,
    quality="Low",
    height=None,
    width=None,
    key=None
)


st.title("Web App Machine Learning")

st.write("""# Explore different classifier and datasets """)

dataset_name = st.sidebar.selectbox(
    "Select dataset", ("Iris", "Breast Cancer", "Wine"))

st.write(f"## {dataset_name} Dataset")


classifier_name = st.sidebar.selectbox(
    "Select classifier", ("KNN", "SVM", "Random Forest"))


def get_dataset(name):
    data = None
    if name == "Iris":
        data = datasets.load_iris()
    elif name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()

    X = data.data
    y = data.target
    return X, y


X, y = get_dataset(dataset_name)
st.write("Shape of dataset: ", X.shape)
st.write("Number of classes: ", len(np.unique(y)))


def add_parameter(clf_name):
    params = dict()
    if clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        params["max_depth"] = max_depth
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["n_estimators"] = n_estimators
    return params


params = add_parameter(classifier_name)


def get_classifier(clf_name, params):
    clf = None
    if clf_name == "SVM":
        clf = SVC(C=params["C"])
    elif clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                           max_depth=params["max_depth"], random_state=1234)
    return clf


clf = get_classifier(classifier_name, params)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


acc = accuracy_score(y_test, y_pred)

st.write(f"Classifier ={classifier_name}")
st.write(f"Accuracy =", acc)


pca = PCA(2)

X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
            c=y, alpha=0.8,
            cmap="Spectral")

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()


st.pyplot(fig)
