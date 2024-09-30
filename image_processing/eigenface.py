import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def run_eigenface(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels)

    # print(np.mean(X_train, axis=0).shape)
    # X_train = X_train - np.mean(X_train, axis=0)
    # X_test = X_test - np.mean(X_train, axis=0)

    pca = PCA().fit(X_train)
    values = np.where(pca.explained_variance_ratio_.cumsum() > 0.95)
    print(values[0][0])
    pca = PCA(n_components=values[0][0]).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)


    X_train_pca = X_train_pca - np.mean(X_train_pca, axis=0)
    X_test_pca = X_test_pca - np.mean(X_train_pca, axis=0)

    print(X_train_pca.shape)
    print(X_test_pca.shape)
    print(y_train.shape)
    print(y_test.shape)

    classifier = SVC().fit(X_train_pca, y_train)
    predictions = classifier.predict(X_test_pca)
    print(predictions.shape)
    print(classification_report(y_test, predictions))