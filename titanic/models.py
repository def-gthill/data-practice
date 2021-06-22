import pandas as pd
from sklearn import dummy
from sklearn import metrics

from bobs import learn

import data


def dummy_model():
    return learn.load_or_train('models/dummy.pkl', train_dummy_model)


def train_dummy_model():
    model = dummy.DummyClassifier()
    model.fit(data.train_x, data.train_y)
    return model


def evaluate(model):
    return metrics.accuracy_score(data.test_y, model.predict(data.test_x))


def submit(fname, model):
    predictions = model.predict(data.submit_x)
    submission = pd.DataFrame(
        {
            "PassengerId": data.submit_x.PassengerId,
            "Survived": predictions,
        }
    )
    submission.to_csv(f"submissions/{fname}", index=False)
