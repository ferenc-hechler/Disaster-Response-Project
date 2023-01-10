import json
import plotly
import pandas as pd

import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import plotly.graph_objs as grob
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

from word_counter import WordCounter


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
# engine = create_engine('sqlite:///../data/YourDatabaseName.db')
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # remove non categorical columns (0..3)
    dfcat=df[df.columns[4:]]
    # count number of occurrences
    cat_count_series = dfcat.sum()
    # limit to the top ten (sort descending, cut)
    cat_count_series = cat_count_series.sort_values(ascending=False).iloc[0:10]
    # ectract category name and values
    cat_counts = cat_count_series.values.tolist()
    cat_names = cat_count_series.index.tolist()

    # create visuals
    graphs = []

    fig1 = [
        grob.Pie(
            labels = cat_names,
            values = cat_counts,
            name = "Categories"
        )
    ]
    lay1 = dict(title = 'Top 10 Categories',
                    xaxis = dict(title = 'Category'),
                    yaxis = dict(title = 'Count'),
                )
    graphs.append(dict(data=fig1, layout=lay1))
    
    graphs.append(
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
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
    )
        
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