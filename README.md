# IMDb Movie Review Sentiment Classifier

## This project implements a Sentiment Analysis model using a Long Short-Term Memory (LSTM) neural network to classify IMDb movie reviews as positive or negative. The model is trained on a dataset of 50,000 labeled movie reviews from the IMDb dataset, which contains an equal distribution of both sentiments.

### Text preprocessing is handled using Keras Tokenizer, which converts raw text into sequences of integers. These sequences are then padded to a fixed length and fed into an embedding layer, followed by an LSTM layer, and finally a dense output layer with sigmoid activation for binary classification.

### The model achieves high accuracy and is deployed using Streamlit for a clean and interactive web-based interface. Users can input a movie review and instantly receive a sentiment prediction along with visual feedback.

