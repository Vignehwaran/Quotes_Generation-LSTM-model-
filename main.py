import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import time

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the pre-trained LSTM model
model = load_model('Quote_completion.h5')


# Set the title of the Streamlit app
st.title("LSTM Text Quote Generator QG-1 own model")
st.success("1 Million Parameter model with 94% accuracy")
 
# Input text from the user
input_text = st.text_input("Enter the beginning of a sentence:", "I Truly")

no_of_words = st.number_input("Number of words to generate:", min_value=1, max_value=50)

st.sidebar.info("this model is trained on 1 million parameters and has an accuracy of 94% . It is trained with 2000 Quotes from the internet. It is a custom model ")
# In your main script where you want the download button (usually at the bottom)
with st.sidebar:
    with open("Quote_completion.h5", "rb") as f:
        st.download_button(
            label="⬇️ Download Trained Model 1M(4MB)",
            data=f,
            file_name="Quote_completion.h5",
            mime="application/octet-stream"
        )
st.sidebar.success("example Quote")
st.sidebar.info("1.There are no problems we cannot solve together, and very few that we can solve by ourselves")
st.sidebar.info("2.Without forgiveness, there's no future.")
st.sidebar.info("3.if you want others to be happy, practice compassion; if you want to be happy, practice compassion.")

# Button to generate text
# if st.button("Generate Quote"):
#     for i in range(no_of_words):
#         token_text=tokenizer.texts_to_sequences([input_text])[0]
#         padded_input=pad_sequences([token_text],maxlen=115,padding='pre') 
#         pos=np.argmax(model.predict(padded_input))
#         for word,index in tokenizer.word_index.items():
#             if index==pos:
#                 input_text=input_text +" "+word
#                 st.write(input_text)
#                 time.sleep(2) 



if st.button("Generate Quote"):
    with st.spinner("Generating quote..."):
        generated_text = input_text
        placeholder = st.empty()
        for i in range(no_of_words):
            token_text = tokenizer.texts_to_sequences([generated_text])[0]
            padded_input = pad_sequences([token_text], maxlen=115, padding='pre')

            pos = np.argmax(model.predict(padded_input))  # Predict next word index
            
            # Find the word corresponding to predicted index
            for word, index in tokenizer.word_index.items():
                if index == pos:
                    generated_text += " " + word  # Append next word to sentence
                    break
            
            placeholder.write(generated_text + "|")  # Show the full sentence updating in one place
            time.sleep(0.1)  # For animation effect

