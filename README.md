# Disaster Response Pipeline Project

## Setup Environment

The packages listed in requirements.txt have to be installed:

```
pip install -r requirements.txt
```

The program was tested with Python 3.10 but it should also run with earlier versions too.


## Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
      the training takes some time (about 10 minutes)

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Start a local browser and enter the url [http://localhost:3000/](http://localhost:3000/).
   Initially the top 10 categories with their percetage occurrences and the distribution of 
   the genres is shown.
   When entering an example sentence, like "we need fresh water" is entred in the input
   field and the button "Classify Message" is clicked a screen showing all identified
   categories in green is shown.


## Running Unittests

For running the unit tests the package "pytest" is needed.

```
cd models
pytest
```

## Sources

The source code is available in GitHub: [Disaster-Response-Project](https://github.com/ferenc-hechler/Disaster-Response-Project).

