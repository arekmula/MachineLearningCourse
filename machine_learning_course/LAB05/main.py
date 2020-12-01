from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import ensemble
from sklearn import impute
from sklearn import metrics
from sklearn import model_selection
from sklearn import svm


def get_diabetes_dataset():
    X, y = datasets.fetch_openml('diabetes', as_frame=True, return_X_y=True)

    return X, y


def todo1():
    X, y = datasets.fetch_openml('diabetes', as_frame=True, return_X_y=True)
    # print(X.info())
    # print(X.describe(include="all"))

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    imputer = impute.SimpleImputer(missing_values=0.0, strategy="mean")

    X_train[["mass"]] = imputer.fit_transform(X_train[["mass"]])
    # X["mass"] = imputer.transform(X["mass"])


    isolation_forest = ensemble.IsolationForest(contamination="auto")
    isolation_forest.fit(X_train)
    y_predicted_outliers = isolation_forest.predict(X_test)
    print(y_predicted_outliers)


    plt.figure()
    X_train.boxplot()

    X_train.hist()

    plt.show()

    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_predicted))

    clf_rf = ensemble.RandomForestClassifier()
    clf_rf.fit(X_train, y_train)
    y_predicted_rf = clf_rf.predict(X_test)
    print(metrics.classification_report(y_test, y_predicted_rf))


def todo1v2():
    X, y = get_diabetes_dataset()

    # Show stats of dataset
    print(X.info())
    print(X.describe())

    # Check results of 3 classification methods on raw data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    print("\n SVC:")
    print(metrics.classification_report(y_test, y_predicted))

    clf_rf = ensemble.RandomForestClassifier()
    clf_rf.fit(X_train, y_train)
    y_predicted_rf = clf_rf.predict(X_test)
    print("\n RandomForest:")
    print(metrics.classification_report(y_test, y_predicted_rf))

    # Fill empty values
    imputer = impute.SimpleImputer(missing_values=0.0, strategy="mean")

    X_train_filled = X_train.copy()
    X_test_filled = X_test.copy()
    X_train_filled[["mass"]] = imputer.fit_transform(X_train[["mass"]])
    X_test_filled[["mass"]] = imputer.transform(X_test[["mass"]])

    # Compare classifiers
    clf_svc_filled = svm.SVC()
    clf_svc_filled.fit(X_train_filled, y_train)
    y_predicted_filled = clf_svc_filled.predict(X_test)
    print(metrics.classification_report(y_test, y_predicted_filled))

    clf_rf_filled = ensemble.RandomForestClassifier()
    clf_rf_filled.fit(X_train, y_train)
    y_predicted_rf_filled = clf_rf_filled.predict(X_test)
    print(metrics.classification_report(y_test, y_predicted_rf_filled))

    # Visualize data
    fig, ax = plt.subplots(2, 1)
    X_train.boxplot(ax=ax[0])
    ax[0].set_title("Surowe dane")
    X_train_filled.boxplot(ax=ax[1])
    ax[1].set_title("Dane po wyczyszczeniu mass")
    plt.show()


# todo1()
todo1v2()