import streamlit as st
import pandas as pd
from recommend import dramasRecommendation 

st.set_page_config(
    page_title="Korean Drama Recommendation System", 
    page_icon="🎬",
    # layout="wide",
    # initial_sidebar_state="expanded"
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/jessicapriscillaimmanuel',
        'Report a bug': "https://www.linkedin.com/in/jessicapriscillaimmanuel",
        'About': "# Jessica Priscilla Immanuel"
    }
)

# Load dataset
data = pd.read_csv('KoreanDramasDatasets.csv')

st.title("Korean Drama Recommendation System")

# Input user
title = st.text_input("Enter a Drama Title You Like:")

filters = st.multiselect(
    "Choose Filters:",
    options=['Genre', 'Actors', 'Rating', 'Description']
)

top_n = st.slider("How many recommendations do you want?", 5, 50, 10)

if st.button("Recommend"):
        if not title:
            st.warning("Please input a drama title first!")
        else:
            recommendations, error = dramasRecommendation(data, title, filters, top_n=top_n)

            if error:
                st.error(error)
            else:
                st.success(f"Here are your top {top_n} recommendations!")
                st.dataframe(recommendations)