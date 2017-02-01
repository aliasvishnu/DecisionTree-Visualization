from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

iris = load_iris()

# Pick 50 test examples at random
testID = np.random.choice(150, 50)

trainX = np.delete(iris.data, testID, axis = 0)
trainY = np.delete(iris.target, testID)

testX = iris.data[testID]
testY = iris.target[testID]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(trainX, trainY)

prediction = clf.predict(testX)
print prediction
print testY

correct = [1 if a == b else 0 for (a, b) in zip(prediction, testY)]

print "Accuracy = ", np.sum(correct)*100.0/len(correct)

# Visualizing the tree
from IPython.display import Image  
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())  