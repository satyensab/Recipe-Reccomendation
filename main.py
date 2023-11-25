# Import neccessary libraries
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast


# Ingredient Image Model Prediction
def model_prediction(ing_image):
    class_label_arr = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
                       'chicken', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes',
                       'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas',
                       'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
                       'sweetpotato', 'tomato', 'turnip', 'watermelon']
    # Load Model
    model = tf.keras.models.load_model("ing_detection_model.h5")

    # Upload Image and convert to image array
    ing_image = load_img(ing_image, target_size=(500, 500))
    img_arr = img_to_array(ing_image)
    img_arr = np.array([img_arr])

    img_arr = img_arr / 255.0  # Normalization

    # Predict Label
    pred = model.predict(img_arr)
    return class_label_arr[np.argmax(pred)]

lemmatizer = WordNetLemmatizer()

def pre_process(ing):
    ing_arr = ing.split(", ")
    cleaned_ing = [lemmatizer.lemmatize(ing) for ing in ing_arr]
    ing_str = ', '.join(cleaned_ing)
    return ing_str
    
# Recommendation Function
def recommend_recipe(ingredients, datframe):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=lambda x: x.split(', '))
    ing_data = vectorizer.fit_transform(datframe['Cleaned_Ingredients'])

    cleaned_ing = pre_process(ingredients)
    input_vec = vectorizer.transform([cleaned_ing])
    
    similarity = cosine_similarity(input_vec, ing_data).flatten()
    top_idx = np.argpartition(similarity, -5)[-5:]

    datframe = datframe.rename(columns={'Title': "Recipe Name"})
    datframe['Ingredients'] = datframe['Ingredients'].apply(lambda x: ast.literal_eval(x))

    rec_df = datframe[['Recipe Name', 'Ingredients', 'Instructions']].iloc[top_idx]
    return rec_df


st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Recipe Reccomendation App", "About Project"])

st.image("app_background.jpg", width=600)

if app_mode == "Recipe Reccomendation App":
    st.header("Recipe Recommendation App 🥪")
    st.text("Not sure what to make? Take a picture or list the ingredients you have in your")
    st.text("pantry and we will give you some recipes you can make!")
    st.text("")
    user_choice = st.radio("Choose an option:", ("List Ingredients ✔️", "Upload Ingredients 📷 (Experimental)"))

    ing_input = ""
    if user_choice == "Upload Ingredients 📷 (Experimental)":
        test_image = st.file_uploader("Upload Ingredients: ", accept_multiple_files=True)
        ingredients_list = []

        # Add ingredients to the ingredients list
        for image in test_image:
            pred = model_prediction(image)
            ingredients_list.append(pred)

        # Display/Label current ingredients list
        if st.button("Current Ingredients List"):
            for image in test_image:
                st.image(image, width=150)
                pred = model_prediction(image)
                st.text(pred)
            if len(ingredients_list) == 0:
                st.text("No images uploaded!")
        ing_input = ', '.join(ingredients_list)
    elif user_choice == "List Ingredients ✔️":
        ing_input = st.text_input("List ingredients (Seperate by a comma. No need to include quantity/measurements!) ",
                                  value="chicken, rice, peas")
    st.text("")
    if st.button("Get Recipes!"):
        df = pd.read_csv("Recipe Dataset.csv")
        df = df.dropna(subset=['Cleaned_Ingredients'])
        rec = recommend_recipe(ing_input, df)
        st.dataframe(rec, hide_index=True, width=1000)

elif app_mode == "About Project":
    st.header("About Project")
    st.text("Based on ingredients provided by user, it will look through 13,500 recipes and whichever ingredients are "
            "the most similar, the corresponding recipe is returned")
    st.text("Datasets used:")
    st.text(
        "✔️ Food Ingredients and Recipes Dataset: https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and"
        "-recipe-dataset-with-images ")
    st.text("✔️ Food Image Dataset: https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition")
    st.text("Resources")
    st.text("✔️ https://www.youtube.com/watch?v=k4IgfyUaW20&list=PLvz5lCwTgdXByZ_z-LFo4vJbbFIMPhkkM&index=11")
    st.text("✔️ https://www.youtube.com/watch?v=eyEabQRBMQA&t=1433s")
    st.text("")
    st.text("(For Ingredient Image Recognition)")
    st.text("The following food items are valid:")
    st.code("fruits - banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango")
    st.code(
        "vegetables - cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, "
        "spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, "
        "jalepeño, ginger, garlic, peas, eggplant.")
    st.code("meats - chicken")
    st.text("More to come!")
