import pandas as pd 
import numpy as np
import locale
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob


class LaptopRecommendationSystem:
    
    def __init__(self, path_to_csv_file):
        self.df = pd.read_csv(path_to_csv_file)