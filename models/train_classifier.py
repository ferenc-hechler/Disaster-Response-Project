import sys
import re
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from word_counter import WordCounter

nltk.download('punkt')


def load_data(database_filepath):
    '''
    read in preprocessed data from given database.
    Calls tokenize for each element.

    Input:
    name of the database file

    Output:
        X: the message column
        Y: all categorical columns (4..)
        category_names: list of category names 
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterMessages', engine)
    X = df["message"]
    category_names = df.columns[4:].tolist()
    Y = df[category_names]
    return X, Y, category_names


def tokenize(text):
    '''
    Uses scikitlearns word_tokenizer to split the given
    text into words (tokens).

    Input:
    string or iterable list of strings

    Output:
    list of words or list of list of words
    '''
    if isinstance(text, str):
        alphanumtext = re.sub("[^a-z0-9]+", " ", text.lower())
        words = word_tokenize(alphanumtext)
        return words
    return [tokenize(sentence) for sentence in text]


def build_model():
    '''
    Create a NLP Pipeline for predicting message categories.
    The initial parameters used here were derived by running 
    GridSearchCV. 

    Output:
    model which can be trained with fit() and then be used 
    for predicting message categories. 
    '''
    vect = CountVectorizer()
    tfidf = TfidfTransformer(smooth_idf=True)
    clf = MultiOutputClassifier(RandomForestClassifier(min_samples_split=2, n_estimators=40))
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('nlp_pipeline', Pipeline([
                ('vect', vect),
                ('tfidf', tfidf)
            ])) ,
            ('word_cnt', WordCounter())
        ])),
        ('clf', clf)
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Use the given model to make a prediction for X_test.
    Output a detailed report about accurracy, recall and 
    F1 score for each given category.
    
    Input:
        model: trained model used for prediction
        X_test: test data
        Y_test: expected result (truth)
        category_names: column names for Y_test 
    '''
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f'COLUMN: {col}')
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    persist model inot model_filepath.
    
    Input:
        model: model to be persisted
        model_filepath: filename, where the model should be stored 
    '''
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


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