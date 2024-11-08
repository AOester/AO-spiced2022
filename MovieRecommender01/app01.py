import streamlit as st
import pandas as pd
from st_aggrid import AgGrid
from recommenders import recomm_movies, random_rate

ratings = pd.read_csv('data01/ratings.csv', index_col=0)
movies = pd.read_csv('data01/movies.csv', index_col=0)
querys = dict(zip(ratings.loc[[1],].movieId ,ratings.loc[[1],].rating))
recommender = 'pop'

st.set_page_config(page_title='Movie recommender on Streamlit' , layout='wide')
st.title('Movie recommender on Streamlit')
bewertung = 0
bewertung = {}
counter = 0


@st.cache
def randomlist():
    rate = random_rate(ratings,movies,numberofratings = 50, amount = 12)
    return rate

rate12 = randomlist()
#st.write(rate12)
#AgGrid(rate12)

#st.table(data=rate12)
st.write('Please rate as many of the following movies as you can.')
st.write('From bad (0.5) to good (5), stepsize 0.5')
st.write('or "no rating" (0).')
st.write(rate12)
#st.write(rate12.iloc[0]['title'])
#st.write(rate12.index)
#st.write(rate12.index[0])


for i in rate12.index:
    bewertung[i] = st.number_input(rate12.iloc[counter]['title'], min_value= 0.0, max_value=5.0, step= 0.5, format=None, key = i)
    counter = counter + 1

#with st.form(key='my_form'):
#	text_input = st.text_input(label='Enter some text')
#	submit_button = st.form_submit_button(label='Submit')


st.write(bewertung)
st.write(type(bewertung))