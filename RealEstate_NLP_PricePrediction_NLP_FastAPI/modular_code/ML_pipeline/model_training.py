# import libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def split_train_test(df, target_variable, size, seed):
    '''df: dataframe
       target_variable: target feature name
       size: test size ratio
       seed: random state'''
    try:
        X = df.drop(target_variable,axis=1)
        y = df[[target_variable]]
        x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=size,random_state=seed)
    except Exception as e:
        print(e)
    else:
        return x_train, x_test, y_train, y_test

def evaluation_matrices(y_train, y_pred_train, y_test, y_pred_test):
    MAE_train = round(mean_absolute_error(y_train,y_pred_train),6)
    MAE_test = round(mean_absolute_error(y_test,y_pred_test),6)
    #Root Mean Squared Error or RMSE
    RMSE_train = round(mean_squared_error(y_train,y_pred_train,squared=False),6)
    RMSE_test = round(mean_squared_error(y_test,y_pred_test,squared=False),6)
    #R2
    R2_train = round(r2_score(y_train, y_pred_train),6)
    R2_test = round(r2_score(y_test, y_pred_test),6)

    return MAE_train, MAE_test, RMSE_train, RMSE_test, R2_train, R2_test

def regression_model_trainer(x_train, x_test, y_train, y_test):
    '''
    Script to train linear regression and regularization models
    :param x_train: training split
    :param y_train: training target vector
    :param x_test: test split
    :param y_test: test target vector
    :return: DataFrame of model evaluation, model objects'''
    try:
        models = [
            ('Linear_Reg', LinearRegression()), 
            ('ridge', Ridge()),
            ('lasso', Lasso())
            ]
        output = pd.DataFrame()
        comparison_columns = ['Model','Train_R2', 'Train_MAE', 'Train_RMSE', 'Test_R2', 'Test_MAE', 'Test_RMSE']
        estimators = []
        for name, model in models:

            rgr = model.fit(x_train, y_train)
            y_pred_train = rgr.predict(x_train)
            y_pred_test = rgr.predict(x_test)
            #Mean Absolute Error or MAE

            estimators.append(rgr)
            MAE_train, MAE_test, RMSE_train, RMSE_test, R2_train, R2_test = evaluation_matrices(y_train, y_pred_train, y_test, y_pred_test)
            
            metric_scores = [name, R2_train, MAE_train, RMSE_train, R2_test, MAE_test, RMSE_test]
            final_dict = dict(zip(comparison_columns,metric_scores))
            df_dictionary = pd.DataFrame([final_dict])
            output = pd.concat([output, df_dictionary], ignore_index=True)
    except Exception as e:
        print(e)
    else:
        return output, estimators[0], estimators[1], estimators[2]
    
    
    
def ensemble_regressor(x_train, x_test, y_train, y_test, estimators):
    '''
    Script to train a voting regressor
    estimators: List of tuples of name and fitted regressor objects'''
    try:
        comparison_columns = ['Model','Train_R2', 'Train_MAE', 'Train_RMSE', 'Test_R2', 'Test_MAE', 'Test_RMSE']
        # train
        voting_ensemble = VotingRegressor(estimators,)
        voting_ensemble.fit(x_train,y_train)

        # predict
        y_pred_train = voting_ensemble.predict(x_train)
        y_pred_test = voting_ensemble.predict(x_test)

        MAE_train, MAE_test, RMSE_train, RMSE_test, R2_train, R2_test = evaluation_matrices(y_train, y_pred_train, y_test, y_pred_test)

        # comparison dataframe
        metric_scores = ['Voting_Ensemble', R2_train, MAE_train, RMSE_train, R2_test, MAE_test, RMSE_test]
        final_dict = dict(zip(comparison_columns,metric_scores))
        df_dictionary = pd.DataFrame([final_dict])
        
    except Exception as e:
        print(e)
    else:
        return df_dictionary, voting_ensemble


