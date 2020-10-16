# -*- coding: utf-8 -*-

"""
Created on Wed Oct 14 16:33:56 2020

@author: Moumita Kamal
"""
from functions import *

def main():
    # importing necessary packages
    from pprint import pprint
    
    import pandas as pd
    
    file_attr = 'Data/' + str(input("Enter file name of the attribute description: "))# 'ids-attr.txt'
    df_attr = pd.read_csv(file_attr, delimiter = ' ')
    df_attr = df_attr.transpose()
    header = list(df_attr.index[[0]])
    header = header+(list(df_attr.iloc[0]))                                         # getting attribute names from 'ids-attr.txt'
    
    file_train = 'Data/' + str(input("Enter file name of the training set: "))      # 'ids-train.txt'
    dataset = pd.read_csv(file_train, names = header, delimiter = ' ')
    class_name = header[len(header)-1]                                              #'class'
    
    file_test = 'Data/' + str(input("Enter file name of the test set: "))           # 'ids-test.txt'
    
                                      
    
    
    
    choice = input('Select an output you wish to see: \n1. Train data set \n2. Test data set \n\nEnter 1 or 2:  ')
    if choice == 1:
        train_data, test_data = functions.train_test_split(dataset)                           # calling train_test_split (optional)
    else:
        train_data = dataset
        test_data = pd.read_csv(file_test, names = header, delimiter = ' ')
    
    tree = functions.Decision_Tree_Learning(train_data, train_data.columns[:-1], None, class_name)          # build tree
    print('\n')
    pprint(tree)                                                                    # print tree
    
    functions.fit(test_data, tree, class_name)                                                            # fit classifier and get predictions
    
    if __name__ == "__main__":
        main()