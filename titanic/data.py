import pandas as pd
from sklearn import model_selection as ms

from bobs import prep

data = pd.read_csv("data/train.csv")
train, test = ms.train_test_split(data, test_size=0.2, random_state=210622)
train_x, train_y = prep.split_y(train, "Survived")
test_x, test_y = prep.split_y(test, "Survived")
submit_x = pd.read_csv("data/test.csv")
