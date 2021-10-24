# decisision tree is an algorithm to classiffy given data
    # it uses two parameters to verify if a given variable
    # can be used as a pivot or a branching point
    # these two parameters are 
        # 1. gini index
        # 2. percentage gain
# https://www.kaggle.com/uciml/pima-indians-diabetes-database
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics # metrics is used to calculate accuracy of the model
import pandas as pd

dataset = pd.read_csv('~/dataset/diabetes.csv')
print(dataset.head())
columns = list(dataset.columns)

target_column = 'Outcome'
Y = dataset[target_column]
columns.remove(target_column)
X = dataset[columns]

tree = DecisionTreeClassifier()
tree.fit(X,Y)
y_pred = tree.predict(X)
print(metrics.accuracy_score(y_pred,Y))
