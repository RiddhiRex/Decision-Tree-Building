# Decision-Tree-Building
Decision tree building

In this project, I have added a max_depth parameter to the decision tree because when the p-value threshold is set to 1, building the entire tree involves including the attributes that are irrelevant which may result in over-fitting, thereby reducing accuracy to almost 65%.


P_value threshold = 0.05
Max_depth	No. of internal nodes	No. of leaves	Accuracy(%)
Complete tree	103	413	71.56


P_value threshold = 0.01
Max_depth	No. of internal nodes	No. of leaves	Accuracy(%)
Complete tree	103	413	74.62


We can see that for P = 1, as the depth increases, the accuracy keeps reducing.
We also observe that the P-value threshold set to 0.01 gives the highest accuracy in this case. This is because when we compute the p-value of the attribute, the attributes that have a very low p-value are more alike to the parent data and thereby contributing more towards the decision at that node. The attributes having a high p-value are not explored further as they are classified as irrelevant.
