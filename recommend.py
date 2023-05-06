import pandas as pd 
import numpy as np
import locale
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob


class LaptopRecommendationSystem:
    
    def __init__(self, path_to_csv_file):
        self.df = pd.read_csv(path_to_csv_file)

    def filter_data(self, brand, color=None, battery_life=None, hard_disk_size=None, max_price=None):
        filtered_data = self.df[self.df['Brand'] == brand]  

        if color:
            filtered_data = filtered_data[filtered_data['Colours'] == color] 

        if battery_life:
            filtered_data = filtered_data[filtered_data['Battery_Life'] == battery_life]  

        if hard_disk_size:
            filtered_data = filtered_data[filtered_data['Hard disk'] == hard_disk_size]  

        if max_price:
            filtered_data['Price in India'] = filtered_data['Price in India'].apply(
                lambda x: float(str(x).replace(",", "")))
            filtered_data = filtered_data[filtered_data['Price in India'] <= max_price]

        return filtered_data
    


    def perform_sentiment_analysis(self, df):
        sentiment_scores = {}
        for index, row in df.iterrows():
            review = row['reviewDescription']
            sentiment = TextBlob(review).sentiment
            sentiment_scores[index] = sentiment

        sorted_scores = sorted(sentiment_scores.items(), key=lambda x: x[1].polarity, reverse=True)
        top_reviews = sorted_scores[:50]  
        
        recommended_items = []
        for review in top_reviews:
            index, sentiment = review
            laptop_model = df.loc[index, 'Model']
            laptop_price = df.loc[index, 'Price in India']
            laptop_colour = df.loc[index, 'Colours']
            laptop_bestbuylink = df.loc[index, 'link']
            laptop_ram = df.loc[index, 'RAM']
            laptop_size = df.loc[index, 'Size']

            recommended_items.append((laptop_model, laptop_price, laptop_colour, laptop_bestbuylink, laptop_ram, laptop_size))

        return recommended_items

    
    def perform_collaborative_filtering(self, df):
        ratings_matrix = df[['1 stars', '2 stars', '3 stars', '4 stars', '5 stars']].values

        item_similarity = cosine_similarity(ratings_matrix.T)

        target_ratings = ratings_matrix.mean(axis=0)

        weighted_ratings = np.dot(item_similarity.T, target_ratings) / np.sum(item_similarity, axis=1)

        sorted_indices = np.argsort(weighted_ratings)[::-1]

        return sorted_indices