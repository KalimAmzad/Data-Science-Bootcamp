# importing libraries
import pandas as pd
import numpy as np
from ML_pipeline import utils
import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

# NLP data preprocessing and feature extraction helper functions
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')

# helper functions to prepare the data

# function to drop null values from dataframe
def drop_null(df):
    try:
        data = df.dropna().reset_index(drop=True)
    except Exception as e:
        print(e)
    else:
        return data

# Function to drop columns
def drop_columns(df, collist):
    try:
        df.drop(collist, axis=1, inplace = True)
    except Exception as e:
        print(e)
    else:
        return df

# function to strip data from a col and save in new col
def strip_data(df, col, newcol, idx):
    '''function to strip data from an object column and save it in a new column of the dataframe
    df: df is the dataframe to be processed
    col: column on which stripping has to be done
    newcol: name of the new column
    idx: stripping index for the value to be stripped
    '''
    try:
        df[newcol] = df[col].apply(lambda x: x.split(',')[idx].lower().strip())
    except Exception as e:
        print(e)
    else:
        return df

# function to seperate numbers out
def strip_numbers(df, col):
    '''The function strips the numerical data from a column
    df: dataframe
    col: column'''
    try:
        newcol = col + ' Cleaned'
        numbers = re.compile(r"[-+]?(\d*\.\d+|\d+)") 
        df[newcol] = df[col].apply(lambda x: numbers.findall(x)[0] if len(numbers.findall(x)) > 0 else 0)
    except Exception as e:
        print(e)
    else:
        return df

# function to clean categorical or textual data case inconsistencies
def clean_text(df, col):
    '''The function removes case inconsistencies from categorical/textual data
    df: dataframe
    col: column'''
    try:
        newcol = col + ' Cleaned'
        df[newcol] = df[col].apply(lambda x:  x.lower().strip())
    except Exception as e:
        print(e)
    else:
        return df

def clean_numbers(df, col):
    '''The function removes numerical inconsistencies
    df: dataframe
    col: column'''
    try:
        newcol = col + ' Cleaned'
        numbers = re.compile(r"[-+]?(\d*\.\d+|\d+)") 
        df[newcol] = (df[col].apply(lambda x: np.float(numbers.findall(str(x))[0]) 
                                                           if len(numbers.findall(str(x)))>0 else np.nan ))
    except Exception as e:
        print(e)
    else:
        return df

# function to encode binary features
def binary_encoder(df, col, mapdict):
    '''The function creates binary encoded features for categorical data
    df: dataframe
    col: column
    mapdict: mapping dictionary'''
    try:
        newcol = col + ' Cleaned'
        df[newcol] = (df[col].apply(lambda x: x.lower().strip()).map(mapdict))
    except Exception as e:
        print(e)
    else:
        return df

# function to calculate average property area (feature cleaning)
def avg_area_calculator(df, col):
    '''The function calculates the average area of the property by first stripping out the numbers from the area ranges provided
    df: dataframe
    col: column'''

    def avg_property_area(x):
        # find numbers from the pattern
        try:
            numbers = re.compile(r"[-+]?(\d*\.\d+|\d+)") 
            x = numbers.findall(x)
            # if a single number is given, return that as area
            if len(x) == 1:
                return np.float(x[0])
            # if a range is given, calculate average and return that as area
            elif len(x) == 2:
                return (np.float(x[0])+np.float(x[1]))/2
            else:
                return -99
        except Exception as e:
            print(e)
    try:
        newcol = col + ' Cleaned'
        df[newcol] = df[col].apply(lambda x: avg_property_area(str(x)))
    except Exception as e:
        print(e)
    else:
        return df

# Function to treat the outliers
def outlier_treatment(df, cols_to_treat):
    # Outlier treatment
    def clip_outliers(df,col):
        try:
            q_l = df[col].min()
            q_h = df[col].quantile(0.95)
            df[col] = df[col].clip(lower = q_l, upper = q_h)
            return df
        except Exception as e:
            print(e)
    try:
        for col in cols_to_treat:
            df = clip_outliers(df,col)
    except Exception as e:
        print(e)
    else:
        return df


# Function to calculate row wise sum of columns
def sum_of_cols(df, col_list, newcol):
    try:
        temp = df[col_list]
        temp[newcol] = temp.sum(axis=1)
        df[newcol] = temp[newcol]
    except Exception as e:
        print(e)
    else:
        return df



# Text cleaning
# Preprocessing the text data
REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;!]")
BAD_SYMBOLS_RE = re.compile("[^0-9a-z #+_]")
STOPWORDS_nlp = set(stopwords.words('english'))

#Custom Stoplist
stoplist = ["i","project","living","home",'apartment',"pune","me","my","myself","we","our","ours","ourselves","you","you're","you've","you'll","you'd","your",
            "yours","yourself","yourselves","he","him","his","himself","she","she's","her","hers","herself","it",
            "it's","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","that'll",
            "these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did",
            "doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about",
            "against","between","into","through","during","before","after","above","below","to","from","up","down","in","out",
            "on","off","over","under","again","further","then","once","here","there","when","where","why","all","any",
            "both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too",
            "very","s","t","can","will","just","don","don't","should","should've","now","d","ll","m","o","re","ve","y","ain",
            "aren","couldn","didn","doesn","hadn","hasn",
            "haven","isn","ma","mightn","mustn","needn","shan","shan't",
            "shouldn","wasn","weren","won","rt","rt","qt","for",
            "the","with","in","of","and","its","it","this","i","have","has","would","could","you","a","an",
            "be","am","can","edushopper","will","to","on","is","by","ive","im","your","we","are","at","as","any","ebay","thank","hello","know",
            "need","want","look","hi","sorry","http", "https","body","dear","hello","hi","thanks","sir","tomorrow","sent","send","see","there","welcome","what","well","us"]

STOPWORDS_nlp.update(stoplist)

# Function to preprocess the text
def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    try:
        text = text.replace("\d+"," ") # removing digits
        text = re.sub(r"(?:\@|https?\://)\S+", "", text) #removing mentions and urls
        text = text.lower() # lowercase text
        text =  re.sub('[0-9]+', '', text)
        text = REPLACE_BY_SPACE_RE.sub(" ", text) # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = BAD_SYMBOLS_RE.sub(" ", text) # delete symbols which are in BAD_SYMBOLS_RE from text
        text = ' '.join([word for word in text.split() if word not in STOPWORDS_nlp]) # delete stopwors from text
        text = text.strip()
        return text
    except Exception as e:
        print(e)

# Pos counter
def pos_counter(x,pos):
    """
    Returns the count for the given parts of speech tag
    
    NN - Noun
    VB - Verb
    JJ - Adjective
    RB - Adverb
    """
    try:
        tokens = nltk.word_tokenize(x.lower())
        tokens = [word for word in tokens if word not in STOPWORDS_nlp]
        text = nltk.Text(tokens)
        tags = nltk.pos_tag(text)
        counts = Counter(tag for word,tag in tags)
        return counts[pos]
    except Exception as e:
        print(e)

# Function to count vectorize text data and combine it to the original dataframe using ngrams

def count_vectorize(df,textcol,ngrams,max_features):
    '''df: Dataframe
    textcol: Text column to be vectorized
    ngrams: ngram_range e.g (2,2)
    max_features: number of maximum most frequent features'''
    try:
        cv = CountVectorizer(ngram_range=ngrams,max_features=max_features)
        # cv = CountVectorizer()
        cv_object = cv.fit(df[textcol])
        X = cv_object.transform(df[textcol])
        df_ngram = pd.DataFrame(X.toarray(),columns=cv_object.get_feature_names())
        # Adding this to the main dataframe
        df_final = pd.concat([df.reset_index(drop=True),df_ngram.reset_index(drop=True)],axis=1)
    except Exception as e:
        print(e)
    else:
        return df_final, cv_object



# Function combine the preprocessing steps for the problem statement

def preprocess_data(df):
    '''The function returns a clean and preprocessed dataset ready to use for training purposes
       Input:
        --df: Raw data dataframe
       
       Output: 
        --df1: Processed Dataframe'''

    try:
        # preprocessing data 
        # Stripping Location details
        for idx, col in enumerate(['City', 'State', 'Country']):
            df = strip_data(df,'Location', col, idx)
            
        # Strip numbers for property type
        df = strip_numbers(df, 'Propert Type')

        # Cleaning text columns
        for col in ['Sub-Area', 'Company Name', 'TownShip Name/ Society Name', 'Description']:
            df = clean_text(df, col)
            
        # Cleaning and encoding Binary Features
        for col in ['ClubHouse','School / University in Township ', 'Hospital in TownShip', 'Mall in TownShip', 'Park / Jogging track',
                'Swimming Pool','Gym']:
            df = binary_encoder(df, col, {'yes':1, 'no':0})
            
        # Cleaning numerical feature: Avg area
        df = avg_area_calculator(df, 'Property Area in Sq. Ft.')

        # Dropping null values
        df = drop_null(df)
        df = clean_numbers(df, 'Price in lakhs')

        # Dropping unnecessary columns
        features = df.columns.tolist()[18:]
        df1 = df[features]

        # Treating outliers in the numeric columns
        cols_to_treat = ['Property Area in Sq. Ft. Cleaned','Price in lakhs Cleaned']
        df1 = outlier_treatment(df1, cols_to_treat)





        # feature engineering and extraction

        # Saving the mapping dict for inference use
        sub_area_price_map = df1.groupby('Sub-Area Cleaned')['Price in lakhs Cleaned'].mean().to_dict()
        utils.pickle_dump(sub_area_price_map,'output/sub_area_price_map.pkl')

        # creating the price by sub-area feature
        df1['Price by sub-area'] =  df1.groupby('Sub-Area Cleaned')['Price in lakhs Cleaned'].transform('mean')

        # amenities col
        amenities_col = df1.columns.tolist()[8:15]
        df1 = sum_of_cols(df1, amenities_col, 'Amenities score')

        # Saving the mapping dict for inference use
        amenities_score_price_map = df1.groupby('Amenities score')['Price in lakhs Cleaned'].mean().to_dict()
        utils.pickle_dump(amenities_score_price_map,'output/amenities_score_price_map.pkl')

        # creating the price by amenities score feature
        df1['Price by Amenities score'] =  df1.groupby('Amenities score')['Price in lakhs Cleaned'].transform('mean')





        # cleaning the description column and creating pos features
        df1["Description Cleaned"] =  df1["Description Cleaned"].astype(str).apply(text_prepare)
        df1['Noun_Counts'] = df1['Description Cleaned'].apply(lambda x: pos_counter(x,'NN'))
        df1['Verb_Counts'] = df1['Description Cleaned'].apply(lambda x: (pos_counter(x,'VB')+ pos_counter(x,'RB')))
        df1['Adjective_Counts'] = df1['Description Cleaned'].apply(lambda x: pos_counter(x,'JJ'))

        # creating count vectors
        df1,cv_object = count_vectorize(df1, 'Description Cleaned',(2,2), 10)

        # dump cv_object for inference purposes
        utils.pickle_dump(cv_object,'output/count_vectorizer.pkl')

        # final df
        # Selecting only numerical features
        cols_to_drop = ['City','State','Country','Sub-Area Cleaned','TownShip Name/ Society Name Cleaned',
                        'Description Cleaned','Company Name Cleaned']
        df1 = drop_columns(df1,cols_to_drop)






        #change feature names, dump the object for inference purposes
        features = list(df1.columns)

        featuresMod = ['PropertyType',
                        'ClubHouse',
                        'School_University_in_Township',
                        'Hospital_in_TownShip',
                        'Mall_in_TownShip',
                        'Park_Jogging_track',
                        'Swimming_Pool',
                        'Gym',
                        'Property_Area_in_Sq_Ft',
                        'Price_in_lakhs',
                        'Price_by_sub_area',
                        'Amenities_score',
                        'Price_by_Amenities_score',
                        'Noun_Counts',
                        'Verb_Counts',
                        'Adjective_Counts',
                        'boasts_elegant',
                        'elegant_towers',
                        'every_day',
                        'great_community',
                        'mantra_gold',
                        'offering_bedroom',
                        'quality_specification',
                        'stories_offering',
                        'towers_stories',
                        'world_class']

        # dump object for inference purposes
        utils.pickle_dump(dict(zip(features,featuresMod)),'output/raw_features_mapping.pkl')
        utils.pickle_dump(featuresMod,'output/features.pkl')
        df1.columns = featuresMod
    
    except Exception as e:
        print(e)
    
    else:
        return df1