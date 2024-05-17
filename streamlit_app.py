import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# Load the model
model = tf.keras.models.load_model('drug_interaction_model.h5')

# Load the LabelEncoders
with open('le_name.pkl', 'rb') as file:
    le_name = pickle.load(file)
with open('le_interaction.pkl', 'rb') as file:
    le_interaction = pickle.load(file)

def check_interaction(drug1_name, drug2_name):
    drug1_id = le_name.transform([drug1_name])[0]
    drug2_id = le_name.transform([drug2_name])[0]
    interaction = model.predict(np.array([[drug1_id, drug2_id]]))
    return interaction[0][0]

# Streamlit UI
st.title("Drug Interaction Checker")

drug1_name = st.text_input("Enter the first drug name:")
drug2_name = st.text_input("Enter the second drug name:")

if st.button("Check Interaction"):
    if drug1_name and drug2_name:
        interaction_score = check_interaction(drug1_name, drug2_name)
        interaction_type = le_interaction.inverse_transform([int(round(interaction_score))])[0]
        st.write(f"Interaction type between {drug1_name} and {drug2_name}: {interaction_type}")
    else:
        st.write("Please enter both drug names to check for interaction.")

