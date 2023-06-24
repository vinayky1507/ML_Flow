import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score
from sklearn.model_selection import train_test_split
import argparse

def get_data():
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    #reading data as df
    df=pd.read_csv(url,sep=";")
    return df

def evaluate(y_true,y_pred):
    mae=mean_absolute_error(y_true,y_pred)
    mse=mean_squared_error(y_true,y_pred)
    rmse=np.sqrt(mean_squared_error(y_true,y_pred))
    r2=r2_score(y_true,y_pred)
    return mae,mse,rmse,r2
def main(n_estimator,max_depth):
    df=get_data()
    print(df)
    train,test=train_test_split(df)
    X_train=train.drop(["quality"],axis=1)
    X_test=test.drop(["quality"],axis=1)
    y_train=train["quality"]
    y_test=test["quality"]

    # lr=ElasticNet()
    # lr.fit(X_train,y_train)
    # pred=lr.predict(X_test)
    # mae,mse,rmse,r2=evaluate(y_test,pred)
    # print(mae,mse,rmse,r2)

    rf=RandomForestClassifier(n_estimators=n_estimator,max_depth=max_depth)
    rf.fit(X_train,y_train)
    pred=rf.predict(X_test)

    accuracy=accuracy_score(y_test,pred)
    print(f"Accurancy score for for rf  {accuracy}")




print("hello bro")
if __name__ == "__main__":

    args=argparse.ArgumentParser()
    args.add_argument("--n_estimator","-n",default=150,type=str)
    args.add_argument("--max_depth","-m",default=15,type=str)
    parse_args=args.parse_args()
    print(parse_args)
    try:
        main(n_estimator=parse_args.n_estimator,max_depth=parse_args.max_depth)
    except Exception as e:
        raise e
