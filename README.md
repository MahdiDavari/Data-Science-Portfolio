# Data Science Portfolio

Repository containing portfolio of data science projects completed by me for academic, self learning, and hobby purposes.
Presented in the form of iPython Notebooks, and Python codes.

>Note: Data used in the projects (accessed under data directory) is for demonstration purposes only.

## Content
- ### Data Science

    - [Data Science Project 1: Incidents responded to by fire companies in NYC in an interactive
     map](https://github.com/MahdiDavari/Data-Science-Portfolio/blob/master/DS1_Interactive_Map/Incidents_Responded_to_by_Fire_Companies_Interactive_Map.ipynb):
    This interactive map show the number of incidents happended in a neighborhood of NYC and the popup
     on any datapoint in a given zip code, give the top 4 incident happened in the zip code during 2013-2017.

    - Here is the link to see the [**HTML Map**](http://www.mdavari.com/Data-Science-Portfolio/DS1_Interactive_Map/Incidents_Responded_to_by_Fire_Companies_Interactive_Map.html).

    - In this project, Incidents responded to by fire companies in NYC dataset is used from [NYC OpenDate](    https://data.cityofnewyork.us/Public-Safety/Incidents-Responded-to-by-Fire-Companies/tm6d-hbzd/data
), Census Populations by Zip Code is used from [Data.gov](https://catalog.data.gov/dataset/2010-census-populations-by-zip-code) and zip code latitude and longitude is used from [Census.gov](https://www.census.gov/geo/maps-data/data/gazetteer2017.html).  
    
  _Tools: Pandas, Folium (Maps visualization), Numpy_

    - [Data Science Project 2: Risk and Returns (The Sharpe Ratio)](https://github.com/MahdiDavari/Data-Science-Portfolio/blob/master/DS2_Risk%20and%20Returns/Risk%20and%20Returns:%20The%20Sharpe%20Ratio.ipynb):
      The Sharpe ratio has been one of the most popular risk/return measures in finance.
      In this project, we calculated the Sharpe ratio for the stocks of the two tech giants
      Facebook and Amazon. As a benchmark, we'll use the S&P 500 that measures the performance
      of the 500 largest stocks in the US.
    
  _Tools: Pandas, Numpy_
    
- ### Machine Learning

    - [Supervised Learning: Machine Learning multi-class, multi-label classifier 
with Random Forest](https://github.com/MahdiDavari/Data-Science-Portfolio/blob/master/ML_Micro_Projects/ML%20with%20Random%20Forest%20(muli-class%2C%20multi-label%20classifier).py):
Random Forest is nothing more than a bunch of Decision Trees combined. They can handle categorical features very well.
This algorithm can handle high dimensional spaces as well as large number of training examples.
Random Forests can almost work out of the box and that is one reason why they are very popular.

   _Tools: scikit-learn, Pandas, Seaborn, Matplotlib, NLTK, Re_
    
    - [Supervised Learning: Phony article detection with Machine Learning (based on its link and title)](https://github.com/MahdiDavari/Data-Science-Portfolio/blob/master/ML_Micro_Projects/ML%20with%20Random%20Forest%20(Article%20classification%20based%20on%20link%20and%20title).ipynb):
A model is trained to detect phony article based on its link and title. Links are parsed with urlparser from urllib library.
 Length of the link, netloc and the title are used as features to train a supervised learning Random Forest classifier model
  to detect whether an article is phony or not.
  
  _Tools: urlparse, scikit-learn, Pandas, NLTK_

    
- ### Natural Language Processing
    - [Text classification using NLTK](https://github.com/MahdiDavari/Data-Science-Portfolio/blob/master/ML_Micro_Projects/ML%20with%20Random%20Forest%20(muli-class%2C%20multi-label%20classifier).py): 
    Using Natural Language Toolkit, I built a classification system for text inputs.
     
   _Tools: scikit-learn, Pandas, Seaborn, Matplotlib, NLTK, Re_
