# decisision tree is an algorithm to classiffy given data
# it uses variation in data to classify
# There is also another attribute which can be used to branch to 
# different decision branches known as the variation gain
# https://www.kaggle.com/itachi9604/disease-symptom-description-dataset
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
