                                Disaster Response Pipeline Project

                                        Introduction
                                        
This project will be working towards understanding the patterns received in disaster data received by Appen (formerly knows as figure 8). Appen providies advance Artificial Intelligence and Machine Learning solutions to its clients. Appen provides services ranging from Data Collection accross a variety of data types such as speech,text, image , video etc., Data Preparation, Model Development and Model Evaluation by both automation and human process. Other services such as Ads Evaluation, Web page evaulation, Catalog-Taxonomy,Related search Content Moderation, and Geo Location Evaluation are some of the other services provided by Appen to its clients to name the few.



Scope of the project
This project is dedicated to build and deploy a reccommendation system which will classify disaster response messages into various categories. The reccommendation system will them assign these classified messeges to the apppropriate disaster management response department for its timely and effective resolution.This project is using data from Figure Eight to create a disaster response pipeline. In order to achieve that, an ETL pipeline was built first to clean and transfer the data. Then a Machine Learning pipeline was created for model building. Lastly, I also created a web app, which including different visualization and message classifier. The message classifier can classify any messages that you typed in. 

Data Understanding
This project will utilize the messages.csv, and categories.csv data files provided by Appen for its analysis.

Two types of models are available to classify the messages. This project utilized Python 3 with libraries of numpy, pandas, sqlalchemy, re, NLTK, pickle, Sklearn, plotly and flask libraries.
1. The first model type is an Adaboost Classifier utilizing Tfidf vectorizer to transform the messages.
2. The second model type that is used for this project is a Random Forest Classifier to transform and classify the messages.




Files used for this project
1. App folder including the templates folder and "run.py" for the web application
2. Data folder containing "DisasterResponse.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py" for data cleaning and transfering.
3. Models folder including "classifier.pkl" and "train_classifier.py" for the Machine Learning model.
4. README file
5. Preparation folder containing 6 different files, which were used for the project building. (Please note: this folder is not necessary for this project to run.)


In order to run and implement this model. Belowmentioned scripts must be ran through command terminal:
- To run ETL pipeline that cleans data and stores in database
 `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
 
 - To run Machine Learning pipeline that trains, classifies and saves model 
 `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
 
 - To run and generate web app
 ` python run.py
 
