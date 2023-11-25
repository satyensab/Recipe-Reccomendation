import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("Recipe Dataset.csv")

df = df.drop(columns="Unnamed: 0")

df['Cleaned_Ingredients'] = df['Cleaned_Ingredients'].apply(lambda x: ast.literal_eval(x))

print(df.shape)

import re
from nltk.corpus import stopwords

custom_stop_words = ['finely', 'chopped', 'small', 'divided', 'plus', 'cups', 'unsalted', 'room', 'temperature', 'cup',
                     'dry', 'about', 'total', 'tablespoons', 'fresh', 'teaspoon', 'cup', 'hot', 'thinly', 'oil']


def extract_ingredient(ingredient):
    # Use stop words common stopwords from NLTK
    stop_words = set(stopwords.words('english'))
    stop_words.update(custom_stop_words)

    original_ing = ingredient

    ingredient = re.sub(r'\([^)]*\)', '', ingredient)
    ingredient = re.split(r',|;|\(', ingredient)[0]
    ingredient = re.sub(r'\S+\s', '', ingredient)

    if not ingredient:
        ingredient = re.sub(r'\([^)]*\)', '', original_ing)
        match = re.search(r'\d[\d\s½¾¼⅓⅔⅛.-]*\s*([^\d(,]+)\s*', ingredient)
        if match:
            ingredient = match.group(1).split(',')[0].strip()

    for word in stop_words:
        cleaned_ingredient = ingredient.replace(word, '')

    return cleaned_ingredient.strip()


df['Cleaned_Ingredients'] = df['Cleaned_Ingredients'].apply(lambda x: [extract_ingredient(ing) for ing in x])

# Find Most Common Ingredients In All Recipes
one_array = []
for ingredients_list in df['Cleaned_Ingredients']:
    for ingredient in ingredients_list:
        one_array.append(ingredient)

all_ingredients = pd.Series(one_array)
most_common_ing = all_ingredients.value_counts(sort=True).head(10)

ingredient_names = most_common_ing.index
plt.bar(most_common_ing.index, most_common_ing)

# Remove the most common ingredients from the recipe
most_common_ingredients = ['salt', 'oil', 'sugar', 'pepper', 'cloves', 'flour', 'juice', '/', 'onion', 'butter']


def remove_most_common_ingredients(ingredients):
    unique_ingredients = [ing for ing in ingredients if ing not in most_common_ingredients]
    return unique_ingredients


df['Cleaned_Ingredients'] = df['Cleaned_Ingredients'].apply(lambda x: remove_most_common_ingredients(x))

# Convert the list of ingredients into a string
df['Cleaned_Ingredients'] = df['Cleaned_Ingredients'].apply(lambda x: ', '.join(x))

# Model Building
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=lambda x: x.split(', '))

ing_data = vectorizer.fit_transform(df['Cleaned_Ingredients'])


# Reccomendation Function
def recommend_recipe(ingredients):
    vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=lambda x: x.split(', '))
    ing_data = vectorizer.fit_transform(df['Cleaned_Ingredients'])

    input_vec = vectorizer.transform([ingredients])
    similarity = cosine_similarity(input_vec, ing_data).flatten()
    top_idx = np.argpartition(similarity, -5)[-5:]
    return df[['Title', 'Ingredients']].iloc[top_idx].values


ing = "cheese, apple, banana"
top_idx = recommend_recipe(ing)
