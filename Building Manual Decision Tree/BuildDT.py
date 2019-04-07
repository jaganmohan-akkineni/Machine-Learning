import sys
import math
import pandas as pd
from collections import Counter
from sklearn.tree import DecisionTreeClassifier


class DecisionNode:

    # A DecisionNode contains an attribute and a dictionary of children. 
    # The attribute is either the attribute being split on, or the predicted label if the node has no children.
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}

    # Visualizes the tree
    def display(self, level = 0):
        if self.children == {}: # reached leaf level
            print(": ", self.attribute, end="")
        else:
            for value in self.children.keys():
                prefix = "\n" + " " * level * 4
                print(prefix, self.attribute, "=", value, end="")
                self.children[value].display(level + 1)
     
    # Predicts the target label for instance x
    def predicts(self, x):
        if self.children == {}: # reached leaf level
            return self.attribute
        value = x[self.attribute]
        subtree = self.children[value]
        return subtree.predicts(x)


# Illustration of functionality of DecisionNode class
def funTree():
    myLeftTree = DecisionNode('humidity')
    myLeftTree.children['normal'] = DecisionNode('no')
    myLeftTree.children['high'] = DecisionNode('yes')
    myTree = DecisionNode('wind')
    myTree.children['weak'] = myLeftTree
    myTree.children['strong'] = DecisionNode('no')
    return myTree


def id3(examples, target, attributes):
    #tree = funTree()
    tree = decisionTree(examples, target, attributes)
   
    return tree

# build the decision tree
def decisionTree(examples, target, attributes):
    bestattribute, baseCase = bestAttribute(examples, target, attributes)
    root= DecisionNode(bestattribute)
    
    if baseCase:
        return root
    
    else:
        attrWithOutBestAttr= attributes.copy()
        attrWithOutBestAttr.remove(bestattribute)
    
        for attrVal in examples[bestattribute].unique():
            dataAttrVal= examples[examples[bestattribute] == attrVal]
            root.children[attrVal]= decisionTree(dataAttrVal, target, attrWithOutBestAttr)
        
        return root


# get the best attribute at each level
def bestAttribute(examples, target, attributes):
    
    targetValues = examples[target].unique()

    if len(attributes) == 0:
        # if attributes is empty, return the single-node tree root with
        # label = most common value of target attribute in examples.
        return (examples[target].value_counts().idxmax(), True)

    elif len(targetValues) > 1:
        entropies = []

        for i in attributes:
            # get current attribute and target value
            attribute = examples[[i, target]]

            attributeValues = attribute[i].unique()

            entropy = getEntropy(examples, target, i, attributeValues, targetValues)

            entropies.append(entropy)

        best_Attribute = attributes[entropies.index(max(entropies))]

        return (best_Attribute, False)

    else:
        return (targetValues[0], True)

def getEntropy(examples, target, attribute, attributeValues, targetValues):
    entropy = 0
    
    df_attribute = examples[[attribute, target]]

    for i in attributeValues:
        df2 = df_attribute.loc[df_attribute[attribute] == i]
        targetValues = df2[target].unique()
        n = df2.shape[0]

        temp_entropy = 0
        counts = df2[str(target)].value_counts().to_dict()
        
        for j in range(targetValues.size):
            if type(targetValues[j]) == type('adf'):
                fraction = (counts.get(str(targetValues[j])))/(df2.shape[0])
            else:
                fraction = (counts.get(int(targetValues[j])))/(df2.shape[0])
            temp_entropy += (fraction) * math.log(fraction)

        entropy += (n/examples.shape[0]) * temp_entropy

    return entropy


####################   MAIN PROGRAM ######################

# Reading input data
#train = pd.read_csv(sys.argv[1])
#test = pd.read_csv(sys.argv[2])
#target = sys.argv[3]
train = pd.read_csv('playtennis_train.csv')
test = pd.read_csv('playtennis_test.csv')
target = 'playtennis'
attributes = train.columns.tolist()
attributes.remove(target)

# Learning and visualizing the tree
tree = id3(train,target,attributes)
tree.display()



# Evaluating the tree on the test data
correct = 0
for i in range(0,len(test)):
    if str(tree.predicts(test.loc[i])) == str(test.loc[i,target]):
        correct += 1
print("\nThe accuracy is: ", correct/len(test))




# Implementing the actual DecisionTreeClassifier from scikit-learn
dtc = DecisionTreeClassifier()

print(train.dtypes)


#mapping classes to labels
outlook_to_int = {'sunny':1, 'overcast':2, 'rainy':3}
train['outlook'] = train['outlook'].replace(outlook_to_int)
test['outlook'] = test['outlook'].replace(outlook_to_int)

temperature_to_int = {'hot':1, 'mild':2, 'cool':3}
train['temperature'] = train['temperature'].replace(temperature_to_int)
test['temperature'] = test['temperature'].replace(temperature_to_int)

humidity_to_int = {'high':1, 'normal':2}
train['humidity'] = train['humidity'].replace(humidity_to_int)
test['humidity'] = test['humidity'].replace(humidity_to_int)

wind_to_int = {'weak':1, 'strong':2}
train['wind'] = train['wind'].replace(wind_to_int)
test['wind'] = test['wind'].replace(wind_to_int)


train_target = train[['playtennis']]
print(train)
train = train.drop(['playtennis'], axis=1)


dtc.fit(train, train_target)
test_target = test['playtennis']
test = test.drop(['playtennis'], axis=1)
print("Score = {}".format(dtc.score(test, test_target)))