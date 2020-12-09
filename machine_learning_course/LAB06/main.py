import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn import datasets, impute
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from matplotlib import pyplot as plt
import numpy as np

titanic = datasets.fetch_openml("Titanic", version="1", as_frame=True)
df_titanic_orig = titanic.data
df_titanic_target = titanic.target


def todo1():
    # print(df_titanic_orig.describe())
    # print(df_titanic_orig.info())
    # print(df_titanic_target)

    # Delete columns boat, body, home.dest
    df_titanic = df_titanic_orig.drop(["boat", "body", "home.dest"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(df_titanic, df_titanic_target, test_size=0.33, random_state=42)

    # Find missing data before filling
    print(X_train.isnull().sum())

    # Fill missing embarked data
    simple_imputer_embarked = impute.SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    X_train_filled = X_train.copy()
    X_train_filled[["embarked"]] = simple_imputer_embarked.fit_transform(X_train[["embarked"]])
    X_test_filled = X_test.copy()
    X_test_filled[["embarked"]] = simple_imputer_embarked.transform(X_test[["embarked"]])

    # Fill mising age data
    plt.figure("Histogram wieku przed uzupelnieniem brakujacych danych")
    plt.hist(X_train_filled["age"])
    iterative_imputer_age = impute.IterativeImputer(missing_values=np.nan, random_state=42)
    X_train_filled[["sibsp", "parch", "age"]] = iterative_imputer_age.fit_transform(X_train_filled[["sibsp",
                                                                                                    "parch",
                                                                                                    "age"]])
    X_test_filled[["sibsp", "parch", "age"]] = iterative_imputer_age.transform(X_test_filled[["sibsp", "parch", "age"]])
    plt.hist(X_train_filled["age"])

    # Drop column "cabin"
    X_train_filled = X_train_filled.drop(["cabin"], axis=1)
    X_test_filled = X_test_filled.drop(["cabin"], axis=1)

    # Fill mising fare data
    iterative_imputer_fare = impute.IterativeImputer(missing_values=np.nan, random_state=42)
    X_train_filled[["pclass", "fare"]] = iterative_imputer_fare.fit_transform(X_train_filled[["pclass", "fare"]])
    X_test_filled[["pclass", "fare"]] = iterative_imputer_fare.transform(X_test_filled[["pclass", "fare"]])

    # Find missing data after filling
    print(X_train_filled.isnull().sum())
    plt.figure("Histogram wieku po uzupelnieniu brakujacych danych")

    # Drop columns name, ticket
    X_train_filled = X_train_filled.drop(["name"], axis=1)
    X_train_filled = X_train_filled.drop(["ticket"], axis=1)
    X_test_filled = X_test_filled.drop(["name"], axis=1)
    X_test_filled = X_test_filled.drop(["ticket"], axis=1)
    # To encode sex, cabin, embarked
    label_encoder_sex = LabelEncoder()
    label_encoder_embarked = LabelEncoder()
    X_train_filled["sex"] = label_encoder_sex.fit_transform(X_train_filled["sex"])
    X_train_filled["embarked"] = label_encoder_embarked.fit_transform(X_train_filled["embarked"])

    X_test_filled["sex"] = label_encoder_sex.transform(X_test_filled["sex"])
    X_test_filled["embarked"] = label_encoder_embarked.transform(X_test_filled["embarked"])

    clf_svc = SVC()
    clf_svc.fit(X_train_filled, y_train)
    print("SVC score: ", clf_svc.score(X_test_filled, y_test))

    clf_rfc = RandomForestClassifier()
    clf_rfc.fit(X_train_filled, y_train)
    print("SVC score: ", clf_rfc.score(X_test_filled, y_test))

    plt.show()


todo1()
