import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
with open('drug_interaction_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the LabelEncoders
with open('le_name.pkl', 'rb') as file:
    le_name = pickle.load(file)
with open('le_interaction.pkl', 'rb') as file:
    le_interaction = pickle.load(file)

def check_interaction(drug1_name, drug2_name):
    try:
        drug1_id = le_name.transform([drug1_name])[0]
    except ValueError:
        st.error(f"Drug name {drug1_name} not recognized.")
        return None
    try:
        drug2_id = le_name.transform([drug2_name])[0]
    except ValueError:
        st.error(f"Drug name {drug2_name} not recognized.")
        return None
    
    id_diff = abs(drug1_id - drug2_id)
    id_sum = drug1_id + drug2_id
    
    # Create DataFrame with the correct feature names
    input_features = pd.DataFrame([[drug1_id, drug2_id, id_diff, id_sum]], 
                                  columns=['drug1_id', 'drug2_id', 'id_diff', 'id_sum'])
    print(input_features)
    interaction = model.predict(input_features)
    return interaction[0]
print(check_interaction("Cyclosporine", "Trimetrexate"))
# # Streamlit UI
# st.title("Drug Interaction Checker")

# drug1_name = st.text_input("Enter the first drug name:")
# drug2_name = st.text_input("Enter the second drug name:")

# if st.button("Check Interaction"):
#     if drug1_name and drug2_name:
#         interaction_type_code = check_interaction(drug1_name, drug2_name)
#         if interaction_type_code is not None:
#             interaction_type = le_interaction.inverse_transform([interaction_type_code])[0]
#             st.write(f"Interaction type between {drug1_name} and {drug2_name}: {interaction_type}")
#     else:
#         st.write("Please enter both drug names to check for interaction.")
