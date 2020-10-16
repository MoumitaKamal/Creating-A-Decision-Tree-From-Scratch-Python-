# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:39:24 2020

@author: Moumita Kamal
"""
import numpy as np
import pandas as pd
from math import log2

class functions:
        
    ##############################################################################################
    # Function: Entropy
    # Parameters: target column of the attribute for which the entropy need to be calculated
    # Description: Calculates entropy of an attribute
    ##############################################################################################

    def Entropy(target_column):
        names, counts = np.unique(target_column, return_counts = True)              # returns value and count of unique members
        entropy = 0
        
        for x in range(0,len(counts)):
            entropy += -(counts[x]/len(target_column)) * log2(counts[x]/len(target_column))
        
        return entropy
    
    
    ##############################################################################################
    # Function: InformationGain
    # Parameters: a set of examples (dataset) and an attribute name 
    # Description: Calculates Information Gain of an attribute with entropy
    ##############################################################################################
    
    def InformationGain(dataset, attribute_name, total_entropy, class_name):
        names, counts = np.unique(dataset[attribute_name], return_counts = True)    # value and count of unique members under that attribute-
        total = len(dataset[attribute_name])                                        
        entropy_wrt_attribute = 0
        
        for x in range(0,len(counts)):
            subset = dataset[class_name].where(dataset[attribute_name]==names[x]).dropna() # subset of examples for a particular child of the attribute
            attribute_entropy = functions.Entropy(subset)                           # entropy for each child
            entropy_wrt_attribute += (counts[x]/total) * attribute_entropy          # total entropy with respect to the attribute
            
        infoGain = total_entropy - entropy_wrt_attribute                            # Information gain
        return infoGain
    
    
    
    ##############################################################################################
    # Function: Choose_Attribute
    # Parameters: a set of examples (dataset) and attributes
    # Description: Calculates the best attribute with Information Gain
    ##############################################################################################
        
    def Choose_Attribute(attributes, dataset, class_name):
        total_entropy = functions.Entropy(dataset[class_name])                      # calculates total entropy for infoGain calculation
        all_infoGain_val = np.array([])
        for x in range(0, len(attributes)):
            all_infoGain_val = np.append(all_infoGain_val, functions.InformationGain(dataset, attributes[x], total_entropy, class_name))
        
        best_index = np.argmax(all_infoGain_val)                                    # returns index of the max value
        best_attribute = attributes[best_index]
        
        return best_attribute
    
    
    
    ##############################################################################################
    # Function: Majority_Value
    # Parameters: a set of examples 
    # Description: Calculates the most common output for the passed in examples
    ##############################################################################################
        
    def Majority_Value(examples, class_name):
         val, c = np.unique(examples[class_name],return_counts=True)                # find count and value of all child
         majority_index = np.argmax(c)                                              # find index of majority value
         
         return val[majority_index]                                                 # return majority value of examples
            
    
    
    ##############################################################################################
    # Function: Decision_Tree_Learning
    # Parameters: a set of examples and attributes, and default value for the goal predicate
    # Description: Calculates the most common output for the passed in examples
    ##############################################################################################
         
    def Decision_Tree_Learning(examples, attributes, default, class_name):
        if len(examples) == 0:
            return default
        elif len(np.unique(examples[class_name])) <= 1:                             # if all examples have the same classification then return the classification
            classification = np.unique(examples[class_name])[0]
            return classification
        elif len(attributes) == 0:
            classification = functions.Majority_Value(examples, class_name)
            return classification
        else:
            best = functions.Choose_Attribute(attributes, examples, class_name)
            tree = {best:{}}
            
            attributes = [i for i in attributes if i != best]                       # remove best from attributes list
                   
            for v in range (0,len(np.unique(examples[best]))):
                value = np.unique(examples[best])[v]
                examples_i = examples.where(examples[best] == value).dropna()
                subtree = functions.Decision_Tree_Learning(examples_i, attributes, functions.Majority_Value(examples, class_name), class_name)
                tree[best][value] = subtree
                
            return (tree)
      
        
        
     ##############################################################################################
    # Function: Predict 
    # Parameters: the generated tree and unlabeled test data 
    # Description: predicts classification for each test sample
    ##############################################################################################
       
    def Predict(X_test, tree):
        for x in list(X_test.keys()):                                               
            if x in list(tree.keys()):                                              # goes through each attribute checking for nodes matching in the tree
                try:
                    prediction = tree[x][X_test[x]] 
                except:
                    return 1                                                        # handles if a certain atribute is absent from the tree
                prediction = tree[x][X_test[x]]
                if isinstance(prediction,dict):                                     # traverses until the right node is found or leaves reached
                    return functions.Predict(X_test,prediction)
    
                else:
                    return prediction
    
        
           
    ##############################################################################################
    # Function: train_test_split (optional)
    # Parameters: a set of examples (dataset) 
    # Description: splits the dataset into train and test sets (75/25)
    ##############################################################################################
    
    def train_test_split(dataset):
        split = int(75/100*len(dataset))                                            # determining index to make the split
        dataset = dataset.sample(frac = 1)                                          # shuffling the dataset
        train = dataset.iloc[:split].reset_index(drop=True)
        test = dataset.iloc[split:].reset_index(drop=True)
        return train, test
    
     
    
    ##############################################################################################
    # Function: peformance
    # Parameters: a set of examples (dataset) and the predictions
    # Description: Calculates and prints the performance metrics
    ##############################################################################################
    
    def peformance(y_pred, dataset, class_name):
        True_pos = 0
        True_neg = 0
        False_pos = 0
        False_neg = 0
        
        for x in range(0, len(y_pred)):
            if y_pred['y_pred'][x] == dataset[class_name][x]:                       # looks for the true predictions               
                if y_pred['y_pred'][x] == 'normal' or y_pred['y_pred'][x] == 'Yes':
                    True_pos += 1
                elif y_pred['y_pred'][x] == 'neptune' or y_pred['y_pred'][x] == 'No':
                    True_neg += 1
            else:                                                                   # looks for the false predictions         
                if y_pred['y_pred'][x] == 'normal' or y_pred['y_pred'][x] == 'Yes':
                    False_pos += 1
                elif y_pred['y_pred'][x] == 'neptune' or y_pred['y_pred'][x] == 'No':
                    False_neg += 1
        
        accuracy = ((True_pos + True_neg)/(True_pos + True_neg + False_pos + False_neg)) * 100
        precision = (True_pos / (True_pos + False_pos)) * 100
        recall = (True_pos / (True_pos + False_neg)) *100
        F1_score = ((2 * (precision * recall)) / (precision + recall))/100
        
        print('\n\nThe prediction accuracy is: ', accuracy,'%')
        print('The precision is: ', precision,'%')
        print('The recall is: ', recall,'%')
        print('The F1_score is: ', F1_score)
    
    
    
    ##############################################################################################
    # Function: fit
    # Parameters: a set of examples (dataset) and the tree
    # Description: fits classifier and gets predictions
    ##############################################################################################
    
    def fit(dataset, tree, class_name):
        X_test = dataset.iloc[:,:-1].to_dict(orient = 'records')                    # getting unlabeled test data (converted to dictionary object since the tree is a dict type)
        y_pred = pd.DataFrame(columns=['y_pred'])                                   # stores predictions
        
        for i in range(len(X_test)):
            y_pred.loc[i,'y_pred'] = functions.Predict(X_test[i],tree)              # gets predictions for each sample of X_test and stored in y_pred
        functions.peformance(y_pred, dataset, class_name)                                       # measures classifier performances
