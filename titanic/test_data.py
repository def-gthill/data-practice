import unittest

import pandas as pd

import data


class TestData(unittest.TestCase):
    def test_train(self):
        self.assertIsInstance(data.train, pd.DataFrame)
        self.assertEqual(
            list(data.train.columns),
            [
                "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
                "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
            ]
        )
        self.assertEqual(891, len(data.train))
