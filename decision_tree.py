#-------------------------------------------------------------------------
# AUTHOR: Anh Tu Nguyen
# FILENAME: decision_tree.py
# SPECIFICATION: Python implementation of ID3 Algorithm
# FOR: CS 4210- Assignment #1
# TIME SPENT: 4.5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#Creating mappings to convert categorical values into numbers
age_mapping = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
spectacle_mapping = {'Myope': 1, 'Hypermetrope': 2}
astigmatism_mapping = {'Yes': 1, 'No': 2}
tear_mapping = {'Reduced': 1, 'Normal': 2}
class_mapping = {'Yes': 1, 'No': 2}

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
for row in db:
  age = age_mapping[row[0]]
  spectacle = spectacle_mapping[row[1]]
  astigmatism = astigmatism_mapping[row[2]]
  tear = tear_mapping[row[3]]
  X.append([age, spectacle, astigmatism, tear])
# X = features

#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2
for row in db:
  Y.append(class_mapping[row[4]])
# Y = class mapping, targer

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()