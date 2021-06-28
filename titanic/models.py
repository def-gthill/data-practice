import pandas as pd
from sklearn import preprocessing as skprep, impute
from sklearn import dummy, linear_model as lm, ensemble
from sklearn import pipeline
from sklearn import metrics

from bobs import prep, learn

import data


def dummy_model():
    return learn.load_or_train("models/dummy.pkl", train_dummy_model)


def train_dummy_model():
    model = dummy.DummyClassifier()
    model.fit(data.train_x, data.train_y)
    return model


def gender_model():
    return learn.load_or_train("models/gender.pkl", train_gender_model)


def train_gender_model():
    model = pipeline.Pipeline([
        ("column", prep.ColumnKeeper(["Sex"])),
        ("onehot", skprep.OneHotEncoder(categories=[["male", "female"]], drop="first")),
        ("classifier", lm.LogisticRegression()),
    ])
    model.fit(data.train_x, data.train_y.values.ravel())
    return model


def full_linear_model():
    return learn.load_or_train("models/full_linear.pkl", train_full_linear_model)


def train_full_linear_model():
    model = pipeline.Pipeline([
        ("preprocessing", standard_prep()),
        ("classifier", lm.LogisticRegressionCV()),
    ])
    model.fit(data.train_x, data.train_y.values.ravel())
    return model


def random_forest_model():
    return learn.load_or_train("models/random_forest.pkl", train_random_forest_model)


def train_random_forest_model():
    features = ["Pclass", "Sex", "SibSp", "Parch"]

    ct = prep.PandasColumnTransformer(
        [
            ("sex", skprep.OneHotEncoder(), ["Sex"]),
        ], remainder="passthrough",
    )
    clf = ensemble.RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    pl = pipeline.Pipeline([
        ("keeper", prep.ColumnKeeper(features)),
        ("transformer", ct),
        ("classifier", clf),
    ])
    pl.fit(data.train_x, data.train_y.values.ravel())
    return pl


def standard_prep():
    numerical = pipeline.Pipeline([
        ("impute", impute.SimpleImputer(strategy="median")),
        ("scale", skprep.StandardScaler())
    ])

    categorical = pipeline.Pipeline([
        ("impute", impute.SimpleImputer(strategy="most_frequent")),
        ("onehot", skprep.OneHotEncoder())
    ])

    ct = prep.PandasColumnTransformer([
        ("numerical", numerical, ["Pclass", "Age", "SibSp", "Parch", "Fare"]),
        ("categorical", categorical, ["Sex", "Embarked"]),
    ])

    return ct


def evaluate(model):
    return metrics.accuracy_score(data.test_y, model.predict(data.test_x))


def coefs(coef_names, coef_values):
    return pd.DataFrame({
        "coef": coef_names,
        "value": coef_values,
    })


def submit(fname, model):
    predictions = model.predict(data.submit_x)
    submission = pd.DataFrame(
        {
            "PassengerId": data.submit_x.PassengerId,
            "Survived": predictions,
        }
    )
    submission.to_csv(f"submissions/{fname}", index=False)
