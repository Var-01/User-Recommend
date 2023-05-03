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