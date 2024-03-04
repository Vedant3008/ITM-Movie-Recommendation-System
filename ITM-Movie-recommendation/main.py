#Have used RFECV for recursive feature extraction with cross validation,it uses a model of SVM
#This process is repeated iteratively until the desired number of features is reached.
#RFECV is useful because it allows you to automatically select the most relevant features in a dataset, which can improve the performance of your model.
#Using this the model we are able to get desired features and then using Joint Probability with Naives Bayes algorithm to get the recommended Movies

#Also used MLE algorithm for the efficiency
import numpy as np
import pandas as pd
from flask import Flask, render_template, request 
#Used flask for deploying on the local web server, easier to routing 
#in home.js and also recommend.js and also to request in AJAX
import bs4 as bs 
#BS4 means beautifulsoup is kind of shortcut to get the data of XML file or html file we will use to get the data from the api
#and we will use to call the api from the TMDB
import urllib.request
 #urllib is used to fetch the data , but we will be using the urllib.request to read and open the URL
# Why are we using it- We are using to read the data from THE IMDB to get and post the reviews of that URL, IMDB provides 
#that option so we added an additional facility 
import pickle
#Pickle module is used to pack the objects in the python, as this project is using the complicated data sets and we were able to find the 
#pkl files for, now pkl files are made using the dump and opened with the help of load disk, it is like it will save the memory as we 
# we have already opened all the list in the pkl files which we downloaded

from datetime import date, datetime
# We are using to check if the movie is released or not

# load the nlp model and tfidf vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))
    
# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

# convert list of numbers to list (eg. "[1,2,3]" to [1,2,3])
def convert_to_list_num(my_list):
    my_list = my_list.split(',')
    my_list[0] = my_list[0].replace("[","")
    my_list[-1] = my_list[-1].replace("]","")
    return my_list

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

app = Flask(__name__)
#Now this python file will run(render) on the home.js 

@app.route("/")#Flask will look up to the static pages like CSS and HTML
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html',suggestions=suggestions)


@app.route("/recommend",methods=["POST"])
def recommend():
    # getting data from AJAX request
    #We have set the request in recommend.js ,We are using the route to POST using AJAX
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    rel_date = request.form['rel_date']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']
    rec_movies_org = request.form['rec_movies_org']
    rec_year = request.form['rec_year']
    rec_vote = request.form['rec_vote']

    # get movie suggestions for auto complete
    suggestions = get_suggestions()

    # call the convert_to_list function for every string that needs to be converted to list
    #There may be a question in Viva why we converted this We converted to string using the above defined function so that
    #In next we can combine list as directories
    #Now there may Arise a question why directories ,because at the end we will be rendoring the directory in HTML and 
    #Why directory-because at the end the whole database in csv at directories only

    rec_movies_org = convert_to_list(rec_movies_org)
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)
    
    # convert string to list (eg. "[1,2,3]" to [1,2,3])
    cast_ids = convert_to_list_num(cast_ids)
    rec_vote = convert_to_list_num(rec_vote)
    rec_year = convert_to_list_num(rec_year)
    

    
    # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
    movie_cards = {rec_posters[i]: [rec_movies[i],rec_movies_org[i],rec_vote[i],rec_year[i]] for i in range(len(rec_posters))}

    casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    # web scraping to get user reviews from IMDB site
    html = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
    soup = bs.BeautifulSoup(html,"html.parser")
    soup_result = soup.find_all("div",{"class":"text show-more__control"})

    #Code snippet copied and pasted from the documentation of the bs4

    reviews_list = [] # list of reviews
    reviews_status = [] # list of comments (good or bad)
    for reviews in soup_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            # passing the review to our model
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Positive' if pred else 'Negative')

    # getting current date(we are using so that we can is it released or not)
    movie_rel_date = ""
    curr_date = ""
    if(rel_date):
        today = str(date.today())
        curr_date = datetime.strptime(today,'%Y-%m-%d')
        movie_rel_date = datetime.strptime(rel_date, '%Y-%m-%d')

    # combining reviews and comments into a dictionary
    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}     

    # passing all the data to the html file
    return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
        vote_count=vote_count,release_date=release_date,movie_rel_date=movie_rel_date,curr_date=curr_date,runtime=runtime,status=status,genres=genres,movie_cards=movie_cards,reviews=movie_reviews,casts=casts,cast_details=cast_details)

if __name__ == '__main__':
    app.run(debug=True) #Flask debug mode
    #Why we used so that we dont have to start the 127 server again and again as we were using Flask Module we thought of using this so that the repetitve process can be avoided
