import os
import sys
from sklearn.metrics import classification_report, roc_auc_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from src.CreditCardDefaultPrediction.utils.utils import load_object



class ModelEvaluation:
    def __init__(self):
        pass

    
    def eval_metrics(self,actual, pred):
        classification_report = classification_report(actual, pred)
        
        roc_auc_score = roc_auc_score(actual, pred)
        return classification_report, roc_auc_score


    def initiate_model_evaluation(self,train_array,test_array):
        try:
            X_test,y_test=(test_array[:,:-1], test_array[:,-1])

            model_path=os.path.join("artifacts","model.pkl")
            model=load_object(model_path)

        

            mlflow.set_registry_uri("https://dagshub.com/pranavbp525/cc_default.mlflow")
            
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            print(tracking_url_type_store)



            with mlflow.start_run():

                predicted_qualities = model.predict(X_test)

                (classification_report, roc_auc_score) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_metric("classification_report", classification_report)
                mlflow.log_metric("roc_auc_score", roc_auc_score)


                # this condition is for the dagshub
                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                # it is for the local 
                else:
                    mlflow.sklearn.log_model(model, "model")


                

            
        except Exception as e:
            raise e