import sys
import pandas as pd
import re
import numpy as np
from sqlalchemy import create_engine, Table, Column, MetaData
import sqlite3
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sklearn.feature_extraction
from sklearn import preprocessing 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target
from sklearn.pipeline import FeatureUnion
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import warnings
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import dill as pickle


database_filepath = 'data/DisasterResponse.db'
model_filepath = 'models/classifier.pkl'
def load_data(database_filepath):
    """ 
    Loads the SQLite databse from the given database_filepath. Stores the messages data in
    df dataframe and further devides the data into input and target variabls.
    Inputs:
          database_filepath-path where the message data stored
    Returns:
           X- inputs containing messages data for modelling
           Y- labels containing the category of messages
           category_names- list of all the messages categories
    """
    
    # Data loading from the database   
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('DisasterResponseMessages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()
   

    return X, Y, category_names

    
def tokenization(text):
    """
    This function will clean and tokenize all the text messages for data modelling.
    The function will also replace all the non-characters with the blank spaces for 
    the model to dervie the exact meaning of the text messages.The function will further
    spilt the strings into words and lemmatized them with Nltk's WordLemmatizer() function
    Input:
          text- list of messages strings that needs to cleaned and tokenized
    Return:
          Cleaned and tokenized list of words
       
    """
    tokens = word_tokenize(text)

    # instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    for strings in text:
        tokens = nltk.word_tokenize(strings)
        lemmatizer = WordNetLemmatizer()
        string_tokenized = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok, pos='v')
            string_tokenized.append(clean_tok.strip())
            
        

    return string_tokenized   


def build_model():
    """
    Builds the pipeline that will transform the messages and the model them based on the user's model selection.
    It will also perform a grid search to find the optimal model parameters.
    INPUTS:
        model_type - the model type selected by the user.
        RETURNS:
        cv_model - the model with the best parameters as determined by grid search
    """
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenization, stop_words=ENGLISH_STOP_WORDS)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    parameters = {'clf__estimator__n_estimators': [10],
                   'clf__estimator__max_features' : ["auto"],
                   'clf__estimator__n_jobs': [4],
                   'clf__estimator__min_samples_split': [2]
                   }
    model = GridSearchCV(pipeline,parameters, cv=3, verbose=2)
    
    pipeline1 = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenization)),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(n_estimators=10,\
                                                             random_state=42)))
        ])
    parameters1 = {'tfidf__ngram_range': ((1, 1),(1,3))
                   }
    ada_model = GridSearchCV(pipeline1,parameters1, cv=3, verbose=1) 
    
    return model
    return ada_model

    


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function will fit and predict precision, f-1, support and accracy scores on test data
    Inputs:
        model # The model that will fit and predict on test data
        X_test # X test values
        y_test # y test values
        category_names # category names
    output:
        scores
    """
   
    y_pred = model.predict(X_test)    
    category_names = Y_test.columns.tolist()
    accuracy_score = (y_pred == Y_test).mean()
    class_report = classification_report(Y_test, y_pred, target_names=category_names)
        
    ## Overall Accuracy Scoring of the model        
    print(accuracy_score,class_report)    
    print({'Random Forest Overall Accuracy Score is':accuracy_score.mean()})     
    

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb')) 
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()