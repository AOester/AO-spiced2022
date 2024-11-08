import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn.decomposition import NMF 
from sklearn.metrics.pairwise import cosine_similarity
import random

# executes a recommender function and the find_movies function
# returns a DataFrame with the recommendations
def recomm_movies(recommender , querys, rating, movies, top=10, filter = 50):

    if recommender == 'pop':
        recommended = recommend_popular(querys, rating, k = top, numberofratings = filter)
    elif recommender == 'ran':
        recommended = random_recommender(querys, rating, k = top, numberofratings = filter)
    elif recommender == 'nmf':
        recommended = nmf_recommender(querys, rating, k = top, numberofratings = filter)
    elif recommender == 'cos':
        recommended = cosine_sim(querys, rating, k = top, numberofratings = filter)
    else :
        return print("choose recommender: 'pop' for popular, 'ran' for random or 'nmf' for nmf")

    return find_movies(recommended , movies)


# getting the movies from the recommender list
# recomm : DataFrame with the recommendet movies
#       index = movieId
#       columns = [['rating' , 'ratesum']]

def find_movies(recomm , movielist):
    to_dict = recomm.reset_index()
    queries = dict(zip(to_dict.loc[:,].movieId ,to_dict.loc[:,].rating))
    choice = movielist.loc[queries.keys()]
    
    return pd.concat([choice,recomm], axis = 1)


def recommender_process(query, ratings, numberofratings = 50):
    # 1. candidate generation
    
    # filter out movies that the user has already seen
    # and calculates the average rating

    # gives a Series where for every movie the mean rating is calculated and already watched movies are dropped
    meanrate = ratings.groupby(['movieId'])['rating'].mean().drop(query.keys())
    # gives a Series where for every movie the number of ratings is calculated and already watched movies are dropped
    # and Series is renamed
    ratesum = ratings.groupby(['movieId'])['rating'].count().drop(query.keys()).rename('ratesum')
    # puts the two above Series in one DataFrame columns are 'rating'(meanrate) and 'ratesum'(sum of rates)
    df = pd.concat([meanrate,ratesum],axis=1)
    # filter out movies that have been watched by less than 20/50/100... users
    # drops movies with less than 'numberofratings' ratings
    df = df.drop(df[df.ratesum < numberofratings ].index )
    return df

# query : Dictionary movieId as index and movie ratings as value
# ratings : DataFrame with a lot of movies with userId as index,
#        columns=[['movieId','rating']]

# collaborative filtering = look at ratings only!
def recommend_popular(query, ratings, numberofratings = 50, k=10): # added how often a movie needs to be rated
    """
    Filters and recommends the top k movies for any given input query. 
    Returns a list of k movie ids.
    """
    df_numb = recommender_process(query, ratings, numberofratings = numberofratings)
    
    # 3. ranking
    # sorts the DataFrame by the mean of rating descending, top k entries are given
    return df_numb.sort_values(by = 'rating',ascending=False)[:k]
    # return the top-k highest rated movie ids


def random_recommender(query, ratings, numberofratings = 50, k = 3):
    
    df_numb = recommender_process(query, ratings, numberofratings = numberofratings)
    
    return shuffle(df_numb)[:k]



def nmf_recommender(query, ratings, numberofratings = 50, k=10):

    reduced = recommender_process(query, ratings, numberofratings)

    tobefitted = ratings.reset_index().pivot(index='userId', columns ='movieId',values='rating').fillna(0)
    movieId = tobefitted.columns.tolist()

    #number of features
    features = 150
    featurelist = []
    for i in range(features):
        featurelist.append(f'feature{i+1}')

    nmf = NMF(n_components=features, init = 'random',max_iter=800)
    nmf.fit(tobefitted)
    Q = pd.DataFrame(nmf.components_, 
                 columns=movieId, 
                 index=featurelist)
    
    P = pd.DataFrame(nmf.transform(tobefitted), 
                 columns=featurelist, 
                 index=tobefitted.index)
    
    recommendations_reconstructed = pd.DataFrame(np.dot(P, Q), 
                                  index=tobefitted.index, 
                                  columns=movieId)

    newID = tobefitted.index.max() +1
    queryframe = pd.DataFrame([query])
    queryframe.index.name = 'userId'
    queryframe.rename(index={0 : newID},inplace = True)

    full = pd.merge(tobefitted.transpose() , queryframe.transpose() , how = 'left', left_index=True,right_index=True).fillna(0).transpose()
    P_new = nmf.transform(full)
    R_new = np.dot(P_new, Q)

    recommendations_new = pd.DataFrame(R_new,
                                index=[full.index],
                                columns=movieId)
    recommendations_new.loc[1].transpose().index.name = 'movieId'
    recom01 = recommendations_new.loc[1].transpose()
    
    dfempfe = pd.merge(reduced , recom01 , how = 'left', left_index=True,right_index=True)
    dfempfe.rename(columns={dfempfe.columns[2]: newID },inplace = True)
    neueempf = dfempfe.sort_values(newID , ascending=False)[:k]
    return neueempf



def cosine_sim(query, ratings, numberofratings = 50, k=10):

    tobefitted = ratings.reset_index().pivot(index='userId', columns ='movieId',values='rating').fillna(0)
    movieId = tobefitted.columns.tolist()
    cosine_sim_table = pd.DataFrame(cosine_similarity(tobefitted), index=tobefitted.index, columns=tobefitted.index)
    tobe_t = tobefitted.T
    active_user = tobe_t.columns[-1]
    unseen_movies = list(tobe_t.index[tobe_t[active_user] == 0])
    neighbours = list(cosine_sim_table[active_user].sort_values(ascending=False).index[0:10])

    predicted_ratings_movies = []

    for movie in unseen_movies:

        num = 0
        den = 0
        
        # we check the users who watched the movie
        people_who_have_seen_the_movie = list(tobe_t.columns[tobe_t.loc[movie] > 0.1])

        for user in neighbours:

            # if this person has seen the movie
            if user in people_who_have_seen_the_movie:
            #  we want extract the ratings and similarities

                rating = tobe_t.loc[movie,user]
                similarity = cosine_sim_table.loc[active_user,user]

            # predict the rating based on the (weighted) average ratings of the neighbours
            # sum(ratings)/no.users OR 
            # sum(ratings*similarity)/sum(similarities)

                num = num + rating*similarity
                den = den + similarity

        if den != 0 :
            predicted_ratings = num/den
            predicted_ratings_movies.append([movie,predicted_ratings])      

    empf = predicted_ratings_movies
    empf = pd.DataFrame(empf).rename({0 : 'movieId', 1 : 'rating'},axis=1).set_index('movieId')

    return empf.sort_values(by = 'rating',ascending=False)[:k]


def random_rate(ratings,movies,numberofratings = 50, amount = 10):
    meanrate = ratings.groupby(['movieId'])['rating'].mean()
    ratesum = ratings.groupby(['movieId'])['rating'].count().rename('ratesum')

    df = pd.concat([meanrate,ratesum],axis=1)
    # filter out movies that have been watched by less than 20/50/100... users
    # drops movies with less than 'numberofratings' ratings
    df = df.drop(df[df.ratesum < numberofratings ].index )

    list = random.sample(range(0, len(df)-1), amount)
    movieliste = movies.iloc[list]
    return movieliste