################################################################################
#                     Confusion Matrix Performance Measure                     #
#                                                                              #
# Perform multiple metric analyses of two .nii images. One image is generated  #
# Using a computer algorithm designed to find the eyeball in the scan, while   #
# the second image is captured by a radiologist. This program then computes    #
# the following metrics; IOU, Dice Coefficient, Haufsdorff distance, MCC and   #
# ACC. This program can process single file pairs, or it can compute metrics   #
# for entire files. Furthermore, this program can out put an overlay of the    #
# two images to visualize their overlap.                                       #
#                                                                              #
# Execution: python performance_measure.py                                     #
# Required Libraries: SimpleITK, numpy, matplotlib. sklearn, csv, os, cv2      #
#                                                                              #
#                                                                              #
################################################################################
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from decimal import *

class measures(object):

    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.matrix = None

    def remove_columns(self):
        self.df.drop(self.df.columns[self.df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        print(self.df.columns)


    def generate_matrix(self):
        self.matrix = confusion_matrix(self.df['Gold Standard'], self.df['Prediction'])
        print(self.matrix)

    def mcc(self):
        matt = matthews_corrcoef(self.df['Gold Standard'], self.df['Prediction'])


def main():
    test1 = measures('Prediction_model_Class_0.csv')
    test1.remove_columns()
    test1.generate_matrix()
    test1.mcc()








    #test1.mcc()

    #class0df = pd.read_csv('Prediction_model_Class_0.csv')
    #print(class0df.columns)

    #class0df.drop(class0df.columns[class0df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    #print(class0df.columns)



    #c_matrix = confusion_matrix(class0df['Gold Standard'], class0df['Prediction'])

    #print(c_matrix)

if __name__=="__main__":
    main()