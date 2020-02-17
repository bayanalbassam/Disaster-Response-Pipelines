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

### Requirements:
To install and run this application, you need following libraries:

Python 3 or above
Databse Libraries: SQLlite and SQLalchemy
Data Analysis Libraries: NumPy and Pandas
Machine Learning Libraries: Scikit-Learn
Natural Language Processing Libraries: NLTK
Web Application Libraries: Flask
Visualization Libraries: Plotly

### Installation:
To install the project on your local machine, clone this repository by running:

git clone https://github.com/bayanalbassam/Disaster-Response-Pipelines.git

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
 https://view6914b2f4-3001.udacity-student-workspaces.com/

### Files Descriptions:
Below is a short description of the main files in this repository:

data/process_data.py: A python script that loads the data from the 'messages.csv' and 'categories.csv' files, merges them, and loads them to a databse file.
models/train_classifier.py:  A python script that creates an AdaBoost machine learning model, tains the model and does a grid-search to find the best model parameter then stores the model in a .pkl file
app/run.py: The main application file, built with Flask framework. It uses the data base files to visualize the message data and the trained ML model to predict an input message importance and categories.


### Acknowledgments:
This project was build as part of Udacity's Data Scientist Nanodegree Program, and it uses code snippets/files from the course. It also uses data from Figure Eight. The support from both Udacity and Figure Eight is greatly acknowledged.
