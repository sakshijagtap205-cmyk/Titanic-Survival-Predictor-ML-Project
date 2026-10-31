from pyexpat import model
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from CaseStudyStep7 import X_test

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
#step 3 = split data
#--------------------------------------------------------------------------------------
#   Function Name =  SplitData
#   Description   =  it perform Spliting Activity
#   Input         =  Dataframe
#   output        =  4 subsets for training and testing
#   Author        =  Sakshi Dattatray Jagtap
#   Date          =  16/08/26
#---------------------------------------------------------------------------------------

def SplitData(df):
    X = df.drop("Survived", axis = 1)
    Y = df["Survived"]
    
    X_train , X_test , Y_train , Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )
    print("Dataset Completed Successfully")
    return X_train , X_test , Y_train , Y_test


# Step 4 = train the model
#--------------------------------------------------------------------------------------
#   Function Name =  train the model
#   Description   =  it performs model training
#   Input         = training features and lables 
#   output        =  trained model
#   Author        =  Sakshi Dattatray Jagtap
#   Date          =  16/08/26
#---------------------------------------------------------------------------------------

def TrainModel(X_train,Y_train):
    model = LogisticRegression(max_iter=1000)
    
    model = model.fit(X_train,Y_train)
    
    print("Model trained successfully")
    
    return model 

# Step 5 = Evaluate the model
#--------------------------------------------------------------------------------------
#   Function Name =  evaluate the model
#   Description   =  it performs model testing
#   Input         =  model , testing data(Features and labels also)
#   output        =  None
#   Author        =  Sakshi Dattatray Jagtap
#   Date          =  16/08/26
#---------------------------------------------------------------------------------------
def EvaluateModel(model, X_test,Y_test):
    Y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(Y_test, Y_pred)
    
    print("Accuracy is:", accuracy)
    print(confusion_matrix(Y_test,Y_pred))
    
# Step 6 = preserve model
#--------------------------------------------------------------------------------------
#   Function Name =  preserve model
#   Description   =  it performs preservation model .pkl file
#   Input         =  model
#   output        =  none
#   Author        =  Sakshi Dattatray Jagtap
#   Date          =  16/08/26
#---------------------------------------------------------------------------------------

def PreserveModel(model,filename):
    joblib.dump(model,filename)
    
    print("Model preserve with name :",filename)
    
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
   
   #step 3 =
   X_train , X_test , Y_train , Y_test = SplitData(df)
   
   # Step 4 = train the model
   model = TrainModel(X_train , Y_train)
   
   #step5 = 
   EvaluateModel(model,X_test,Y_test)
   
   #step 6 =
   PreserveModel(model,"MarvellousTitanic.pkl")
   
if __name__ == "__main__":
    main() 