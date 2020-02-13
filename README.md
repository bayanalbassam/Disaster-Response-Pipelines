 # Disaster-Response-Pipelines
This project is to classify disaster response messages through machine learning


### Content:
1. Data
    - process_data.py: reads the data, cleans it, and stores it in a SQL database.
    - disaster_categories.csv and disaster_messages.csv:  (datasets)
    - DisasterResponse.db: created database from cleaned and transformed data.
2. Models
    - train_classifier.py: includes the code necessary to load data, transform it, run a machine learning model using GridSearchCV and train it.
3. App
    - run.py: Flask app and the user interface used to predict results and display them.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
