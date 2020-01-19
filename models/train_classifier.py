import sys
# import libraries
import pandas as pd
import numpy as np

from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import GridSearchCV

import pickle

def load_data(database_filepath):
    '''Loads data from database and returns list of messages, applicable categories and names of categories'''
  
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    # List of tables in database
    table = engine.table_names()
    # print('Below table/tables are available in database,\n',table[0])
    # Read database table into dataframe
    df = pd.read_sql_table(table[0], engine)
    # Split features and labels
    X = df['message']
    y = df.iloc[:,5:] #All columns from 5th
    # List of category columns
    categories = y.columns
    #print('Data load and splitting features and target finished..')
    return X, y, categories


def tokenize(text):
    '''Tokenize input text and returns tokens'''
    # Detect urls and replace with placeholders
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # Normalise the text
    list_of_processed_words = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize
    list_of_processed_words = word_tokenize(list_of_processed_words)
    # Remove stopwords
    list_of_processed_words = [w for w in list_of_processed_words if w not in stopwords.words("english")]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in list_of_processed_words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

# Below is the function to create a model using grid search but this seems long running
# hence I am not able to check the best parameters and proceeded with model witout grid search with default hyper parameters
# in function named as build_model

def build_CV_model():
    '''Build model using pipeline and gridsearch for best hyper parameters'''
    
    # Create a model classifier
    multi_clf = MultiOutputClassifier(RandomForestClassifier())
    
    # Create a pipeline object with elements which needs to be appiled in training dataset
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', multi_clf)])
    
    # Create a dictionary for input parameters using which model will be trained.
    # I am prooving the default values as training is taking too long for multiple parameters
    parameters = {'clf__estimator__n_estimators': [100,150,200]}
    
    # Create a model using gridsearch and pipeline
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model

def build_model():
    '''Creates a model using RandomForestClassifier using pipeline'''
    multi_clf = MultiOutputClassifier(RandomForestClassifier())
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', multi_clf)
    ])
    return model


    
    
def evaluate_model(model, X_test, y_test, category_names):
    '''Takes model, test features and labels, category names and prints f_score, precision and recall for each category '''
    
    # Store values predicted by model in a list
    y_pred = model.predict(X_test)
    
    # Creating an empty dataframe with below columns
    report_df = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    i = 0
    
    # Calculate precision, recall and f_score for each category and append in dataframe create above
    for column in category_names:
        precision, recall, f_score, support = metrics.precision_recall_fscore_support(y_test[column], y_pred[:,i], 
                                                                              average='weighted')
        #print(column,':',precision, recall, f_score)
        report_df = report_df.append({'Category':column,'f_score':f_score,'precision':precision,
                                     'recall':recall},ignore_index=True)
        i += 1
    # Print detailed report
    print(report_df)



def save_model(model, model_filepath):
    '''Takes model and pickle filename and saves model as provided filename'''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    


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