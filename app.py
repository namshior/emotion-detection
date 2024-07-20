from random import random
from xml.parsers.expat import model
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import numpy as np
import nltk
nltk.download('punkt')
from sklearn.datasets import load_files
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import string 
from nltk.stem import WordNetLemmatizer
import sqlite3
from googletrans import Translator
import warnings
from topic_modelling import Topic_modeling

app = Flask(__name__)

#Reading data
translator = Translator()

df= pd.read_csv("data.csv")
df = df[0:2000]

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
df['Emotion']= label_encoder.fit_transform(df['Emotion'])

Tweet = []
Labels = []

for row in df["Text"]:
    #tokenize words
    words = word_tokenize(row)
    #remove punctuations
    clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
    #remove stop words
    english_stops = set(stopwords.words('english'))
    characters_to_remove = ["''",'``',"rt","https","’","“","”","\u200b","--","n't","'s","...","//t.c" ]
    clean_words = [word for word in clean_words if word not in english_stops]
    clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
    #Lematise words
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
    Tweet.append(lemma_list)



df['message']=df['Text']



#df = df[0:2000]
X = df['message']
y = df['Emotion']

# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data

import joblib
model = joblib.load("model.sav")

from sklearn.ensemble import RandomForestClassifier
NB = RandomForestClassifier()
NB.fit(X,y)
predictions = NB.predict(X)

@app.route('/')
def home():
	return render_template('home.html')

@app.route("/signup")
def signup():
    
    
    name = request.args.get('username','')
    number = request.args.get('number','')
    email = request.args.get('email','')
    password = request.args.get('psw','')

    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `detail` (`name`,`number`,`email`, `password`) VALUES (?, ?, ?, ?)",(name,number,email,password))
    con.commit()
    con.close()

    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('name','')
    password1 = request.args.get('psw','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `name`, `password` from detail where `name` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()
    print(data)

    if data == None:
        return render_template("signup.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")



@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        
        message = request.form['message']
        translations = translator.translate(message, dest='en')
        message =  translations.text
        data = [message]
        #cv = tfidf.fit_transform(
        vect = cv.transform(data).toarray()
        my_prediction = NB.predict(vect)
        
        print(my_prediction[0])
        df = pd.DataFrame({'sentence':data})
        t,word = Topic_modeling(df)
        return render_template('result.html',prediction = my_prediction[0],message=message,to = t, wo = word)


@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/reg')
def reg():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route('/notebook')
def notebook():
	return render_template('Notebook.html')

@app.route('/about')
def about():
	return render_template('about.html')

if __name__ == '__main__':
	app.run(debug=False)
