import csv
from rake_nltk import Rake
import nltk
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import string

#I had already created a bag of words under All words in md2 and loaded that into cleaneddata.csv so I can go straight to actual 
#recommender system in this module and its more organized

def cosinesimilarity(recommendtitle):
    df = pd.read_csv("/Users/sachittarora/Documents/GitHub/level1_post_recommender_20/sachitt_arora/cleaneddata.csv")

   
    recommendations = []
    ival = 0

    count = CountVectorizer()
    countmatrix = count.fit_transform(df['All words'])
    cosinesimilarity = cosine_similarity(countmatrix, countmatrix)

    #find index of the recommend title
    for index, row in df.iterrows():
        if df["Title"][index] == recommendtitle:
            ival = index
    #use index to find the cosinesimilarity of the index
    scores = pd.Series(cosinesimilarity[ival]).sort_values(ascending = False)
    top10 = list(scores.iloc[1:11].index)  
    
    for i in top10:  
        recommendations.append(list(df['Title'])[i])

    
    return recommendations



print(cosinesimilarity("better speaker than stock aiy v1"))

print(cosinesimilarity("mark ii update november 2020"))

print(cosinesimilarity("mycroft as sip client smart voice enabled ivr"))

