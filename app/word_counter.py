import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class WordCounter(BaseEstimator, TransformerMixin):
    '''
    Estimator for the NLP Pipeline which transforms a 
    text column into the new feature "number of words".

    Input:
    dollar_string: input in format '$NNN.NN'

    Output:
    float value for the given input
    '''
        
    def fit(self, X, y=None):
        return self
    
    def tokenize(self, text):
        '''
        Uses scikitlearns word_tokenizer to split the given
        text into words (tokens).
    
        Input:
        string
    
        Output:
        list of words
        '''
        alphanumtext = re.sub("[^a-z0-9]+", " ", text.lower())
        words = word_tokenize(alphanumtext)
        return words
    
    def transform(self, X):
        '''
        Transforms a list of sentences into a list of number of words (numeric).
        Calls tokenize for each element.
    
        Input:
        column with strings
    
        Output:
        numeric dataframe 
        '''
        result = [len(self.tokenize(s)) for s in X]
        return pd.DataFrame(result)
