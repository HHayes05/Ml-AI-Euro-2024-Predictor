Hugh Hayes

To run the prediction run euro_prediction.ipynb

1.  WebScraping to Get Data: 

multiple_years_euro.py scrapes the web to get the fixtures for the 2024 Euros, and also gets the historical results of all previous euros. 

Some data may be missing from the data, so we use get_missing_data.py and selenium to fill in the remaining data. (Not Needed but helpful just incase)

get_dict_table.ipynb gets a dictionairy of the groups and teams in the groups for euro 2024. 

data_cleaning.ipynb is used to clean up the euro_fixture csv and the euro result csv into a more readable format. 


2. Euro Predictor
Euro Predictor Model:
The Euro predictor model is the core of the project, implemented in the euro_prediction.ipynb notebook. Here's a breakdown of its key components and functionality:

Data Loading and Preparation:

Loads the cleaned historical results (cleaned_euro_results.csv) and fixtures (cleaned_euro_fixture.csv) data.
Imports the group tables dictionary (dict_table) for the Euro 2024 tournament.

Machine Learning Models:
Utilizes three Random Forest models from scikit-learn:
a) A classifier to predict match outcomes (win, lose, draw)
b) Two regressors to predict goal scores for home and away teams

Data Processing:
Encodes team names using LabelEncoder for machine learning compatibility.
Prepares features (team names) and target variables (match outcomes and scores) for model training.

Model Training:
Trains the Random Forest models on the historical Euro data.

Match Prediction Function:
Implements a predict_match function that uses the trained models to predict outcomes and scores for individual matches.
Incorporates a Poisson distribution to add realistic variability to the predicted scores.

Tournament Simulation:
Simulates the entire Euro 2024 tournament, including:
a) Group stage matches
b) Updating group tables after each match
c) Determining knockout stage qualifiers
d) Simulating knockout matches up to the final


Results Display:
Prints match results, group standings, and highlights key matches in later stages of the tournament.

Flexibility and Randomness:
Handles teams not in the training data with random predictions.
Balances AI predictions with added randomness for more realistic and varied outcomes.
