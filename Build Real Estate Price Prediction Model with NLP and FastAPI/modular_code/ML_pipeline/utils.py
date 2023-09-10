import pandas as pd
import numpy as np
import pickle
import os
from scipy import stats
import sklearn

# Function to read the data
def read_data(file_path, **kwargs):
    try:
        raw_data = pd.read_excel(file_path  ,**kwargs)
    except Exception as e:
        print(e)
    else:
        return raw_data


# Function to dump python objects
def pickle_dump(data, filename):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename,'wb') as f:
            pickle.dump(data,f)
    except Exception as e:
        print(e)

# Function for prediction interval
def get_prediction_interval(prediction, actual_values, predicted_values, pi=.95):
    '''
    Get a prediction interval for the regression model.
    
    INPUTS: 
        - Single prediction (test data), 
        - y_train
        - prediction from x_train,
        - Prediction interval threshold (default = .95) 
    OUTPUT: 
        - Prediction interval for single test prediction
    '''
    try:
        #get standard deviation of prediction on the train dataset
        sum_errs = np.sum((actual_values - predicted_values)**2)
        stdev = np.sqrt(sum_errs / (len(actual_values) - 1))
        
        #get interval from standard deviation
        one_minus_pi = 1 - pi
        ppf_lookup = 1 - (one_minus_pi / 2) # If we need to calculate a 'Two-tail test' (i.e. We're concerned with values both greater and less than our mean) then we need to split the significance (i.e. our alpha value) because we're still using a calculation method for one-tail. The split in half symbolizes the significance level being appropriated to both tails. A 95% significance level has a 5% alpha; splitting the 5% alpha across both tails returns 2.5%. Taking 2.5% from 100% returns 97.5% as an input for the significance level.
        z_score = stats.norm.ppf(ppf_lookup) # This will return a value (that functions as a 'standard-deviation multiplier') marking where 95% (pi%) of data points would be contained if our data is a normal distribution.
        interval = z_score * stdev
        
        
        #generate prediction interval lower and upper bound cs_24
        lower, upper = prediction - interval, prediction + interval
        return lower[0], upper[0]
    except Exception as e:
        print(e)



def price_ranges(model_obj, x_train, x_test, y_train):
    '''The function takes in fitted model objects and returns a data frame for lower and upper price ranges
        using prediction or confidence interval'''
    # getting prediction intervals for the test data
    try:
        lower_vet = []
        upper_vet = []
        if isinstance(model_obj, sklearn.linear_model.Lasso) == True or isinstance(model_obj, sklearn.ensemble.VotingRegressor):
            preds_test = model_obj.predict(x_test).reshape(-1,1)
        else:
            preds_test = model_obj.predict(x_test)
        for i in preds_test:
            lower, upper =  get_prediction_interval(i, y_train.values, model_obj.predict(x_train).reshape(-1,1))
            lower_vet.append(lower)
            upper_vet.append(upper)
        df = pd.DataFrame(zip(lower_vet,upper_vet, preds_test.reshape(-1).tolist()),columns=['lower','upper','mean'])
        return df
    except Exception as e:
        print(e)




