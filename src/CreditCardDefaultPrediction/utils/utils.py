import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.CreditCardDefaultPrediction.exception import customexception
from src.CreditCardDefaultPrediction.logger import logging

from sklearn.metrics import roc_auc_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)
    
def evaluate_model(X_train_resampled, y_train_resampled,X_test_scaled,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train_resampled, y_train_resampled)

            if hasattr(model, "predict_proba"):
                y_test_pred = model.predict(X_test_scaled)
                y_test_pred_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for the positive class
            else:
                y_test_pred = model.predict(X_test_scaled)
                y_test_pred_prob = y_pred

            

            # Get roc scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = roc_auc_score(y_test,y_test_pred_prob)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise customexception(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise customexception(e,sys)

    