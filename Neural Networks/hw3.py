# course: TCSS555
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score, roc_curve, auc

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)


df = pd.read_csv('BreastCancer.csv')

X = df.drop('diagnosis', axis=1)
map_output = {2:0, 4:1}
y = df.diagnosis
y = y.replace(map_output)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, stratify=y,random_state=45)
####################### simple decision tree ####################### 
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree = clf.predict(X_test)

print("Accuracy: %.2f" % accuracy_score(y_test,y_pred_decision_tree))
y_prob = clf.predict_proba(X_test)
print(clf.classes_)
fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
print("AUC for Decision Tree: %.2f" % roc_auc)
####################### end decision tree ####################### 

####################### simple neural network ####################### 



model = Sequential()

#add your code here
model.add(Dense(5, input_dim=9, activation='sigmoid'))
#model.add(Dense(18, input_dim=9, activation='relu'))
#model.add(Dense(5, input_dim=9, activation='sigmoid'))
#model.add(Dense(4, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
# binary cross entropy is logarithmic loss for binary classification problem
# adam is the gradient descent algorithm
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model(training)
# batch-size is no of instances that are evaluated before each weight update
#model.fit(X, Y, epochs=150, batch_size=10)
model.fit(X_train,y_train,epochs=100)

y_pred_neural_network = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_neural_network[:, 0], pos_label=1)
roc_auc = auc(fpr, tpr)
print("AUC for Neural Network: %.2f" % roc_auc)