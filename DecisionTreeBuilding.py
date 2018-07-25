import random
import pickle as pkl
import argparse
import csv
import scipy.stats
import pandas as pd
import numpy as np

'''
TreeNode represents a node in your decision tree
TreeNode can be:
    - A non-leaf node: 
        - data: contains the feature number this node is using to split the data
        - children[0]-children[4]: Each correspond to one of the values that the feature can take

    - A leaf node:
        - data: 'T' or 'F' 
        - children[0]-children[4]: Doesn't matter, you can leave them the same or cast to None.

'''

# DO NOT CHANGE THIS CLASS
class TreeNode():
    def __init__(self, data='T', children=[-1] * 5):
        self.nodes = list(children)
        self.data = data

    def save_tree(self, filename):
        obj = open(filename, 'wb')
        pkl.dump(self, obj)


# loads Train and Test data
def load_data(ftrain, ftest):
    Xtrain, Ytrain, Xtest = [], [], []
    with open(ftrain, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = map(int, row[0].split())
            Xtrain.append(rw)

    with open(ftest, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = map(int, row[0].split())
            Xtest.append(rw)

    ftrain_label = ftrain.split('.')[0] + '_label.csv'
    with open(ftrain_label, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = int(row[0])
            Ytrain.append(rw)

    print('Data Loading: done')
    return Xtrain, Ytrain, Xtest

#Finds the attribute with highest information gain
def infoGain(TD):
    TD_length = len(TD)
    min_entropy = 99999
    minEntColumn = -1
    for column in TD:
        #Ignoring output column
        if column == 'Y':
            continue
        col_entropy = 0
        for value in set(TD[column]):
            Y_counts = TD.loc[TD[column].isin([value])]['Y'].value_counts()
            sum_Y_counts = sum(Y_counts)
            col_entropy += (float(sum_Y_counts)/float(TD_length)) * scipy.stats.entropy([float(x) / float(sum_Y_counts) for x in Y_counts], base = 2)
        #Find min entropy (or max gain)
        if col_entropy < min_entropy:
            min_entropy = col_entropy
            minEntColumn = column
    return minEntColumn

#Runs the ID3 algorithm to generate a decision tree
def ID3Algo(T, node, depth, best_attrib, p_value):
    Y_counts = T['Y'].value_counts()
    #Depth-limiter
    if depth == 0:
        if Y_counts[1] > Y_counts[0]:
            return TreeNode('T')
        else:
            return TreeNode('F')
    if best_attrib == -1:
            if Y_counts.index[0] == 1:
                return TreeNode('T')
            else:
                return TreeNode('F')
    node.data = best_attrib
    children = [-1,-1,-1,-1,-1]
    child_attrib = {}
    cols = set(T.columns)
    cols.remove(best_attrib)
    T_length = len(T)
    val_list = set(T[best_attrib])
    for value in range(1,6):
        #Making a guess for values not present in training set
        if value not in val_list:
            children[value - 1] = TreeNode('F')
        else:
            #Computing the p-value of each split
            splitData = T.loc[T[best_attrib].isin([value]), cols]
            observed = splitData['Y'].value_counts()
            expected = T['Y'].value_counts() / float(T_length) * float(len(splitData))
            S, p = scipy.stats.chisquare(observed,expected)

            #Comparing with the threshold
            if p > p_value:
                #pure subset case
                if len(observed) == 1:
                    if observed.index[0] == 1:
                        child = TreeNode('T')
                    else:
                        child = TreeNode('F')
                else:
                    if observed[1] > observed[0]:
                        child = TreeNode('T')
                    else:
                        child = TreeNode('F')
            else:
                #pure subset case
                if len(observed) == 1:
                    if observed.index[0] == 1:
                        child = TreeNode('T')
                    else:
                        child = TreeNode('F')
                else:
                    #Compute best attribute inside the split
                    best_attrib_child = infoGain(splitData)
                    #To avoid recomputation of tree at the same level, store the child node in a dict
                    if str(best_attrib_child) in child_attrib:
                        child = child_attrib[str(best_attrib_child)]
                    else:
                        child = ID3Algo(splitData, TreeNode(), depth-1, best_attrib_child, p_value)
                        child_attrib[str(best_attrib_child)] = child
            children[value - 1] = child
    node.nodes = children
    return node

#Computes the output label for a row in test set
def TreeOutputForRow(root, datapoint):
    if root.data == 'T': return 1
    if root.data == 'F': return 0
    return TreeOutputForRow(root.nodes[datapoint[int(root.data) - 1] - 1], datapoint)

#Computes output labels using a decision tree on a test set
def TreeOutput(T, root):
    Ypred = []
    for i in range(0, len(T)):
        Ypred.append([TreeOutputForRow(root, T[i])])
    return Ypred

#Command line arguments parser
parser = argparse.ArgumentParser()
parser.add_argument('-p', required=True)
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)
parser.add_argument('-t', help='output tree filename', required=True)

args = vars(parser.parse_args())

pval = args['p']
Xtrain_name = args['f1']
Ytrain_name = args['f1'].split('.')[0]+ '_labels.csv' #labels filename will be the same as training file name but with _label at the end

Xtest_name = args['f2']
Ytest_predict_name = args['o']

tree_name = args['t']

#Load data
X_train, Y_train, X_test = load_data(Xtrain_name, Xtest_name)
#Convert to pandas dataframe
x_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(Y_train)
#Setting output column to Y
y_train.columns = ['Y']
XY_train = pd.concat([x_train,y_train], axis = 1)

#Initialize parameters to ID3Algo
root = TreeNode()
best_attrib = infoGain(XY_train)
max_depth = 6
decision_tree = ID3Algo(XY_train,root, max_depth, best_attrib, float(pval))
#Save the tree
decision_tree.save_tree(tree_name)
print("Testing...")
#Compute predictions
Ypredict = TreeOutput(X_test, decision_tree)
#Write output labels to file
with open(Ytest_predict_name, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Ypredict)

print("Output files generated")
