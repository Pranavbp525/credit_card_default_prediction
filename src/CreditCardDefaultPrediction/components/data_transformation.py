import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.preprocessing import RobustScaler
from imblearn.combine import SMOTEENN

from src.CreditCardDefaultPrediction.exception import customexception
from src.CreditCardDefaultPrediction.logger import logging

from src.CreditCardDefaultPrediction.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    scaler_obj_file_path=os.path.join('artifacts','scaler.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


            
    
    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("read train and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            sex_map = {1: "male", 2: "female"}
            education_map = {1: "graduate school", 2: "university", 3: "high school", 0: "others", 4: "others", 5: "others", 6: "others"}
            marriage_map = {1: "married", 2: "single", 3: "divorce", 0: "others"}
            pay_map = {-2: "no consumption", -1: "paid in full", 0: "the use of revolving credit", 1: "payment delay for one month", 2: "payment delay for two months",
                    3: "payment delay for three months", 4: "payment delay for four months", 5: "payment delay for five months", 6: "payment delay for six months",
                    7: "payment delay for seven months", 8: "payment delay for eight months", 9: "payment delay for nine months and above"}
            

            logging.info("Mapping the int columns to actual category descriptions")
            train_df['SEX'] = train_df['SEX'].replace(sex_map)
            train_df['EDUCATION'] = train_df['EDUCATION'].replace(education_map)
            train_df['MARRIAGE'] = train_df['MARRIAGE'].replace(marriage_map)
            train_df['PAY_0'] = train_df['PAY_0'].replace(pay_map)
            train_df['PAY_2'] = train_df['PAY_2'].replace(pay_map)
            train_df['PAY_3'] = train_df['PAY_3'].replace(pay_map)
            train_df['PAY_4'] = train_df['PAY_4'].replace(pay_map)
            train_df['PAY_5'] = train_df['PAY_5'].replace(pay_map)
            train_df['PAY_6'] = train_df['PAY_6'].replace(pay_map)
            train_df.drop(columns=['ID'], inplace=True)
            train_df.rename(columns={'default.payment.next.month': 'default'}, inplace=True)

            test_df['SEX'] = test_df['SEX'].replace(sex_map)
            test_df['EDUCATION'] = test_df['EDUCATION'].replace(education_map)
            test_df['MARRIAGE'] = test_df['MARRIAGE'].replace(marriage_map)
            test_df['PAY_0'] = test_df['PAY_0'].replace(pay_map)
            test_df['PAY_2'] = test_df['PAY_2'].replace(pay_map)
            test_df['PAY_3'] = test_df['PAY_3'].replace(pay_map)
            test_df['PAY_4'] = test_df['PAY_4'].replace(pay_map)
            test_df['PAY_5'] = test_df['PAY_5'].replace(pay_map)
            test_df['PAY_6'] = test_df['PAY_6'].replace(pay_map)
            test_df.drop(columns=['ID'], inplace=True)
            test_df.rename(columns={'default.payment.next.month': 'default'}, inplace=True)

            #logging.info("Capping to remove outliers")
            #logging.info("Applying transformation to reduce skewness.")

            test_df = pd.get_dummies(test_df)
            train_df = pd.get_dummies(train_df)

            columns = list(test_df.columns)
            columns.remove('default')
            columns.append('default')
            train_df = train_df[columns]
            test_df = test_df[columns]


            scaler = RobustScaler()
            smote_enn = SMOTEENN(random_state=42)
            
            target_column_name = 'default'
            drop_columns = [target_column_name]
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            input_feature_train_arr_scaled=scaler.fit_transform(input_feature_train_df)
            input_feature_train_arr_resampled, target_feature_train_arr_resampled=smote_enn.fit_resample(input_feature_train_arr_scaled,target_feature_train_df)
            
            input_feature_test_arr_scaled=scaler.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            train_arr = np.c_[input_feature_train_arr_resampled, np.array(target_feature_train_arr_resampled)]
            test_arr = np.c_[input_feature_test_arr_scaled, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.scaler_obj_file_path,
                obj=scaler
            )
            
            logging.info("scaler pickle file saved")
            
            return (
                train_arr,
                test_arr
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise customexception(e,sys)
            
    