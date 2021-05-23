"""
Plotting functions
"""

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def distplot(data, x):
    """
    Recreates the (deprecated) sns.distplot function
    """
    sns.histplot(data=data, x=x, kde=True, stat='density', linewidth=0)


def confusionplot(y_true, y_pred, labels, display_labels=None):
    if display_labels is None:
        display_labels = labels
    confusion = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(
        pd.DataFrame(confusion, index=display_labels, columns=display_labels),
        annot=True, cmap='YlGnBu', fmt='g',
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

