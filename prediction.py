import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

model=load_model("review.h5")
tokenizer=joblib.load("tokenizer.pkl")

def predict (data):
    data=tokenizer.texts_to_sequences([data])
    pad_data=pad_sequences(data , maxlen=200)
    output=model.predict(pad_data)
    sentiment="positive" if output[0][0]>0.5 else "negative"
    return sentiment


import streamlit as st

st.title('IMDb Movie Review Sentiment Analysis')
st.write('Enter a movie review to analyze')
user_input = st.text_area('Movie Review')
if st.button('Classify'):
    output=predict(user_input)
    st.write(f'Sentiment: {output}')
else:
    st.write('Please enter a movie review.')


