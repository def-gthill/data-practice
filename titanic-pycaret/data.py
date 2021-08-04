import pandas as pd
from sklearn import model_selection as ms

data = pd.read_csv("data/train.csv")
train, test = ms.train_test_split(data, test_size=0.2, random_state=210622)
submit_x = pd.read_csv("data/test.csv")
