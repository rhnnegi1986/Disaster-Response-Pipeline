import sys
import pandas as pd
import re
import numpy as np
from sqlalchemy import create_engine, Table, Column, MetaData
import sqlite3

messages_filepath = 'data/disaster_messages.csv'
categories_filepath = 'data/disaster_categories.csv'
database_filename = 'DisasterResponse.db'



def load_data(messages_filepath, categories_filepath):
    """
    Load & merge messages & categories datasets
    
    inputs:
    messages_filepath: string. Filepath for csv file containing messages dataset.
    categories_filepath: string. Filepath for csv file containing categories dataset.
       
    outputs:
    df: dataframe. Dataframe containing merged content of messages & categories datasets.
    """
    
    messages = pd.read_csv('data/disaster_messages.csv', dtype='str')
    categories = pd.read_csv('data/disaster_categories.csv')

    messages = messages.drop('original', axis=1)

    # Converting id data type for the categories df for mergig with the messages datset
    categories['id'] = categories['id'].astype('str')


    # merge datasets
    df = pd.merge(messages,categories,left_on='id',right_on='id')
    
    return df



def clean_data(df):
    """
    Clean dataframe by removing duplicates & converting categories from strings 
    to binary values.
    
    Args:
    df: dataframe. Dataframe containing merged content of messages & categories datasets.
       
    Returns:
    df: dataframe. Dataframe containing cleaned version of input dataframe.
    """
    
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    for column in categories:   
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
        
     # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)


     # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat((df, categories), axis=1)

    df['message'] = df['message'].str.replace('\W',' ')

    # drop duplicates
    df = df[~df.duplicated(keep=False)]

    # changing the id datatype as int
    df['id'] = df['id'].astype('int64')
    # Remove rows with a  value of 2 from df
    df = df[df['related'] != 2]
        
    return df

def save_data(df, database_filename):
    """    
    Save into  SQLite database.
    
    inputs:
    df: dataframe. Dataframe containing cleaned version of merged message and 
    categories data.
    database_filename: string. Filename for output database.
       
    outputs:
    None
    
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponseMessages', engine, index=False, if_exists='replace')

    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
