import os
import warnings
from abc import ABCMeta, abstractmethod

import pandas as pd
from sklearn import preprocessing as skprep, impute
from sklearn import dummy, linear_model as lm, ensemble
from sklearn import pipeline
from sklearn import metrics

from bobs import prep, learn

import data


class Model(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def make_model(self):
        pass

    @abstractmethod
    def make_refit_model(self):
        """
        Makes a model suitable for refitting, i.e.
        one that uses the best hyperparameters already
        found instead of trying to tune them.
        """
        pass

    def save_path(self, variant: str = "") -> str:
        return f"models/{self.name}{'_' + variant if variant else ''}.pkl"

    def clear_saved(self, variant: str = ""):
        """Clears the saved model, forcing it to be re-trained"""
        try:
            os.remove(self.save_path(variant))
        except OSError:
            warnings.warn(f"Can't clear the {self.name} model")

    def get_trained_model(self, variant: str, x, y):
        """
        Loads the saved model if available, otherwise trains
        a new model on the specified x and y values and saves
        it
        """
        def train():
            model = self.make_model()
            model.fit(x, y)
            return model

        return learn.load_or_train(self.save_path(variant), train)

    def get_split_model(self):
        """
        Trains the model on the train data only, or loads
        it if already trained
        """
        return self.get_trained_model(
            "", data.train_x, data.train_y.values.ravel()
        )

    def get_refit_model(self):
        """
        Trains the model on the full data set, or loads
        it if already trained. Do this to get the final
        model after choosing the design that works best
        """
        return self.get_trained_model(
            "refit", data.data_x, data.data_y.values.ravel()
        )


class DummyModel(Model):
    @property
    def name(self) -> str:
        return "dummy"

    def make_model(self):
        return dummy.DummyClassifier()

    def make_refit_model(self):
        return self.make_model()


# def dummy_model():
#     return learn.load_or_train("models/dummy.pkl", train_dummy_model)
#
#
# def train_dummy_model():
#     model = dummy.DummyClassifier()
#     model.fit(data.train_x, data.train_y)
#     return model


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
