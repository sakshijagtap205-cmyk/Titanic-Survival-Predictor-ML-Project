import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1 = load the data
#--------------------------------------------------------------------------------------
#   Function Name =  LoadData
#   Description   =  Load the data from csv
#   Input         =  Name of the csv file
#   output        =  Data Frame
#   Author        =  Sakshi Dattatray Jagtap
#   Date          =  16/08/26
#---------------------------------------------------------------------------------------

def LoadData(filename):
    df=pd.read_csv(filename)
    
    print("Dataset loaded successfully")
    print(df.head())
    
    return df
#--------------------------------------------------------------------------------------
#   Function Name =  LoadData
#   Description   =  Entry point function
#   Input         =  Name of the csv file
#   output        =  Data Frame
#   Author        =  Sakshi Dattatray Jagtap
#   Date          =  16/08/26
#---------------------------------------------------------------------------------------

def main():
    LoadData("MarvellousTitanicDataset.csv")    
if __name__ == "__main__":
    main() 