import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score, make_scorer, accuracy_score
from sklearn.metrics import precision_score, f1_score, recall_score, classification_report
from typing import Tuple, List
import numpy as np
import pickle

nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath: str)->Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load the filepath and return the data
    INPUT: database_filepath(String)
    OUTPUT:
     - X: The messages value array (numpy.ndarray)
     - Y: The categories value (numpy.ndarray)
     - categories: The categories (String list)
    """
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM Messages", engine)
    
    # Create X and Y datasets
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    # Create list containing all category names
    category_names = list(Y.columns.values)
    
    return X, Y, category_names



def tokenize(text:str)->List[str] :
    """
    tokenize and transform input text
    INPUT: The text to be processed (String)
    OUTPUT: Processed tokens (String list)
    """
    tokens = nltk.word_tokenize(text)

    # create a lemmatizer and apply it to each token
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(x).lower().strip() for x in tokens]


def build_model(): 
    """
    Builds the machine learning model estimator
    INPUT:: None
    OUTPUT: Built model estimator (sklearn.model_selection.GridSearchCV)
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier()) 
    ])
    
    parameters = {'clf__max_depth': [10, 40, None],
              'clf__min_samples_leaf':[2, 5, 10],
                 }
    
    cv = GridSearchCV(pipeline, parameters)
    
    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    """
    Print model results
    INPUT:
    model, X_test, y_test, category_names(list of category strings)
    OUTPUT: None
    """
    # Get results and add them to a dataframe.
    # Use the model for prediction
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=category_names))
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])

def save_model(model, model_filepath):
    """Save model as pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))

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
              'train_classifier.py ../data/disaster_response.db classifier.pkl')


if __name__ == '__main__':
    main()
