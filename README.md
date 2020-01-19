# Disaster Response Pipeline Project

 
Analysing disaster data from a data set containing real messages that were sent during disaster events. 
A machine learing model using pipeline to categorize these messages into various categories.
One message can be associated to multiple categories hence a multi-ouput ML model is required.

A WebApp where user can enter any new message to see various possible categories to which that message may belong to, 
on webapp few graphs to analyse training dataset.

Dataset provide by FigureEight (https://www.figure-eight.com/)



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

