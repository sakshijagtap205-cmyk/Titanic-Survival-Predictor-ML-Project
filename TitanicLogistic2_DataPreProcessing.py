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

#step 2 = Data Preprocessing
#--------------------------------------------------------------------------------------
#   Function Name =  PreprocessData
#   Description   =  it perform data analysis
#   Input         =  Dataframe
#   output        =  update Data Frame
#   Author        =  Sakshi Dattatray Jagtap
#   Date          =  16/08/26
#---------------------------------------------------------------------------------------

def PreProcessData(df):
    df = df.drop([
        "Passengerid",
        "zero",
        "name"
    ],
    errors="ignore"
    )
    #Handle Missing Values
    df["Age"]= df["Age"].fillna(df["Age"].mean())
    df["Fare"]= df["Fare"].fillna(df["Fare"].median())
    
    df["Embarked"]= df["Embarked"].fillna(df["Embarked"].mode()[0])
    #convert catagorical newmweric data
    
    df= pd.get_dummies(
        df,
        columns=["Embarked"],
        drop_first=True,
        dtype=int
    )
    
    print(df.head())
    
    print("Data PreProcessing Completed")
    
    return df

#--------------------------------------------------------------------------------------
#   Function Name =  main
#   Description   =  Entry point function
#   Input         =  None
#   output        =  None
#   Author        =  Sakshi Dattatray Jagtap
#   Date          =  16/08/26
#---------------------------------------------------------------------------------------

def main():
    #step 1 =
   df = LoadData("MarvellousTitanicDataset.csv")   
   
   #step 2 =
   df = PreProcessData(df)
    
if __name__ == "__main__":
    main() 