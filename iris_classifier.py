#loading the iris dataset
from sklearn import datasets
iris=datasets.load_iris()

#assign the data and target to separate the variables
#x contains the features and y contains the labels
x=iris.data
y=iris.target

#splitting the dataset
#x_train contains training features, y_train contains training labels
#x_test contains testing features, y_test contains testing labels
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5)

#building the model
from sklearn import tree
classifier=tree.DecisionTreeClassifier()

#train the model
classifier.fit(x_train,y_train)

#make predictions
predictions=classifier.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))