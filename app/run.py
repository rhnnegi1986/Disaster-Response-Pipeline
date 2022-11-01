import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponseMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    percent_gen = round(100 * genre_counts/gen_counts.sum(),2)
    genre_names = list(genre_counts.index)
    number_of_categories = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()
    number_of_categories = number_of_categories.sort_values(ascending = False)
    Total_Categories = list(number_of_categories.index)

    colors = ['blue', 'red', 'green']
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                {"type": "pie",
                 "uid": "f4de1f",
                 "name": "Genre",
                 "domain":{
                     "x": percent_gen,
                     "y": genre_names},
                 "marker":{
                     "colors":[
                         "#7fc97f",
                         "#beaed4",
                         "#fdc086"
                     ]
                 },
                 "textinfo": "label+value",
                 "hoverinfo": "all",
                 "labels": genre_names,
                 "values": genre_counts
                }
            ],
            "layout" :{
                "title": "Number and Messages Percent by Genre Names"
            }
        },
        {
            "data": [
                {
                "type": "bar",
                    "x" : Total_Categories,
                    "y": number_of_Categories,
                    "marker":{
                        "color": 'brown'}
                }
                    ],                  
                    

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
