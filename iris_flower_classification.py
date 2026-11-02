#Iris Flower Classification Using Decision Trees
#Iris dataset contains data about sepal and petal lenghts and widths of 3 types of iris flower

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import plot_tree #function to visualize trained decision tree
import matplotlib.pyplot as plt

#Load the Iris dataset
iris = load_iris() #inlcudes features like sepal length, sepal width, petal length, and petal width

#Create a DataFrame from the Iris datset
df = pd.DataFrame(data=iris.data, columns=iris.feature_names) #converts dataset into pandas dataframe for easier manipulation
df['species'] = iris.target #Add the species column (target)

#Display first few rows of the dataset
print("Iris Dataset:")
print(df.head())

#Split dataset into features(X) and target (y)
X = df.drop('species', axis=1) #Features (sepal length, sepal width, petal length, petal width)
y = df['species'] #Target (species: 0=Setosa, 1=Versicolor, 2=Virginica)

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Train a Decision Tree Classifier
classifier = DecisionTreeClassifier() #initialize classifier
classifier.fit(X_train, y_train) #train classifier on the training set

#Make predicitions on the test set
y_pred = classifier.predict(X_test) #predicts species of the iris flowers in the test set

#Evaluate the model using accuracy, confusion matrix, and classification report
accuracy = metrics.accuracy_score(y_test, y_pred)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_report = metrics.classification_report(y_test, y_pred) #precision,recall,and f1 score

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

#Visualize the Decision Tree
plt.figure(figsize=(12,8))
plot_tree(classifier, feature_names=iris.feature_names, class_names=iris.target_names, filled=True) #filled= True colors the tree nodes based on the class it predicts
plt.title("Decision Tree for Iris Flower Classification")
plt.show()