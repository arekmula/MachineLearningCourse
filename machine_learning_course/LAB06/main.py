import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn import datasets, impute
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
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
    print("RFC score: ", clf_rfc.score(X_test_filled, y_test))

    plt.show()


def todo2():
    # Delete columns boat, body, home.dest
    df_titanic = pd.DataFrame(df_titanic_orig.drop(["boat", "body", "home.dest"], axis=1))
    df_titanic["survived"] = titanic.target

    df_titanic_sex_survival = pd.DataFrame()
    df_titanic_females = df_titanic.loc[df_titanic["sex"] == "female"]
    df_titanic_males = df_titanic.loc[df_titanic["sex"] == "male"]
    df_titanic_sex_survival.loc["female", "survived"] = (df_titanic_females["survived"].value_counts()["1"] /
                                                         len(df_titanic_females))
    df_titanic_sex_survival.loc["female", "dead"] = 1 - df_titanic_sex_survival.loc["female", "survived"]
    df_titanic_sex_survival.loc["male", "survived"] = (df_titanic_males["survived"].value_counts()["1"]
                                                       / len(df_titanic_males))
    df_titanic_sex_survival.loc["male", "dead"] = 1 - df_titanic_sex_survival.loc["male", "survived"]

    print(df_titanic_sex_survival)

    labels = ["Female", "Male"]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x-width/2, df_titanic_sex_survival["survived"], width, label="survived")
    ax.bar(x+width/2, df_titanic_sex_survival["dead"], width, label="dead")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()

    df_titanic_embarked_survival = pd.DataFrame()
    df_titanic_S = df_titanic.loc[df_titanic["embarked"] == "S"]
    df_titanic_C = df_titanic.loc[df_titanic["embarked"] == "C"]
    df_titanic_Q = df_titanic.loc[df_titanic["embarked"] == "Q"]

    df_titanic_embarked_survival.loc["S", "survived"] = (df_titanic_S["survived"].value_counts()["1"] /
                                                         len(df_titanic_S))
    df_titanic_embarked_survival.loc["C", "survived"] = (df_titanic_C["survived"].value_counts()["1"] /
                                                         len(df_titanic_C))
    df_titanic_embarked_survival.loc["Q", "survived"] = (df_titanic_Q["survived"].value_counts()["1"] /
                                                         len(df_titanic_Q))
    print(df_titanic_embarked_survival)


def todo3():
    """
    Use standard scalling to improve classification
    :return:
    """

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
    standard_scaler_fare = StandardScaler()
    standard_scaler_age = StandardScaler()
    X_train_filled["fare"] = standard_scaler_fare.fit_transform(X_train_filled["fare"][:, np.newaxis])
    X_train_filled["age"] = standard_scaler_age.fit_transform(X_train_filled["age"][:, np.newaxis])

    X_test_filled["sex"] = label_encoder_sex.transform(X_test_filled["sex"])
    X_test_filled["embarked"] = label_encoder_embarked.transform(X_test_filled["embarked"])
    X_test_filled["fare"] = standard_scaler_fare.transform(X_test_filled["fare"][:, np.newaxis])
    X_test_filled["age"] = standard_scaler_age.transform(X_test_filled["age"][:, np.newaxis])

    clf_svc = SVC()
    clf_svc.fit(X_train_filled, y_train)
    print("SVC score: ", clf_svc.score(X_test_filled, y_test))

    clf_rfc = RandomForestClassifier()
    clf_rfc.fit(X_train_filled, y_train)
    print("RFC score: ", clf_rfc.score(X_test_filled, y_test))

    plt.show()


def todo4():
    """
    Use automatic feature selection
    :return:
    """
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
    min_max_scaler_fare = MinMaxScaler()
    min_max_scaler_age = MinMaxScaler()
    X_train_filled["fare"] = min_max_scaler_fare.fit_transform(X_train_filled["fare"][:, np.newaxis])
    X_train_filled["age"] = min_max_scaler_age.fit_transform(X_train_filled["age"][:, np.newaxis])

    X_test_filled["sex"] = label_encoder_sex.transform(X_test_filled["sex"])
    X_test_filled["embarked"] = label_encoder_embarked.transform(X_test_filled["embarked"])
    X_test_filled["fare"] = min_max_scaler_fare.transform(X_test_filled["fare"][:, np.newaxis])
    X_test_filled["age"] = min_max_scaler_age.transform(X_test_filled["age"][:, np.newaxis])

    print(f"Ilosc cech przed wyborem: {X_train_filled.shape}")
    select_kbest = SelectKBest(chi2, k=5)
    X_train_filled = select_kbest.fit_transform(X_train_filled, y_train)
    print(f"Ilosc cech po wyborze: {X_train_filled.shape}")
    X_test_filled = select_kbest.transform(X_test_filled)

    clf_svc = SVC()
    clf_svc.fit(X_train_filled, y_train)
    print("SVC score: ", clf_svc.score(X_test_filled, y_test))

    clf_rfc = RandomForestClassifier()
    clf_rfc.fit(X_train_filled, y_train)
    print("RFC score: ", clf_rfc.score(X_test_filled, y_test))

    plt.show()


# todo1()
# todo2()
# todo3()
todo4()

