import math
import unittest

import pandas as pd
import data


class TestData(unittest.TestCase):
    def test_data(self):
        self.assertIsInstance(data.data, pd.DataFrame)
        self.assertEqual(
            list(data.data.columns),
            [
                "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
                "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
            ]
        )
        self.assertEqual(891, len(data.data))

    def test_train_test_split(self):
        self.assertEqual(math.floor(891 * 0.8), len(data.train))
        self.assertEqual(math.ceil(891 * 0.2), len(data.test))
