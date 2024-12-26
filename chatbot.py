import json
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import csv
import datetime
import pathlib
import os

# Load intents file
with open('intents.json') as file:
    data = json.load(file)

# Initialize vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=1000)

# Preprocess the data
tags = []
patterns = []
for intent in data:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    # Predicting the tag
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in data:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that. Can you please clarify?"

# Function to load CSS
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the external CSS
css_path = pathlib.Path("style.css")
load_css(css_path)

def main():
    st.image('image.webp', width=500)
    st.title("CUSTOMER SUPPORT")
    # Creating sidebar options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home page
    if choice == 'Home':
        st.write('Welcome to the customer support, please enter your message here')
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['User Inputs', 'ChatBot Response', 'Timestamp'])
        
        user_input = st.text_input("USER:")

        if user_input:
            response = chatbot(user_input)
            st.text_area("ChatBot:", value=response, height=120, max_chars=None)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([user_input, response, timestamp])
            
            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a good day!")
                st.stop()

    # Conversation history
    elif choice == 'Conversation History':
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    st.text(f'User: {row[0]}')
                    st.text(f'ChatBot: {row[1]}')
                    st.text(f'Timestamp: {row[2]}')
                    st.markdown("---")
        else:
            st.write("No conversation history found.")



    # About page
    elif choice == "About":
         st.header("About")
         st.write("""
    Welcome to our Intent-Based Chatbot!

    This chatbot is designed to provide an interactive and intuitive experience for users by leveraging Natural Language Processing (NLP) and machine learning techniques. Here's a bit more about how it works and what it offers:

    ### How It Works
    Our chatbot utilizes a Logistic Regression model to predict the intent behind the user's input. This model is trained on a dataset of various user queries and responses, allowing it to accurately classify and respond to a wide range of inquiries.

    ### Features
    - *Intent Recognition*: The chatbot can understand and categorize user inputs into predefined intents, ensuring relevant and helpful responses.
    - *Dynamic Responses*: Based on the recognized intent, the chatbot generates appropriate responses, making the interaction seamless and engaging.
    - *User-Friendly Interface*: Implemented using Streamlit, the chatbot provides a clean and intuitive web interface, making it easy for users to interact and get the information they need.

    ### Technologies Used
    - *Natural Language Processing (NLP)*: To process and understand the text input from users.
    - *Logistic Regression*: A machine learning algorithm used to classify the intents.
    - *Streamlit*: A powerful tool for creating and deploying web applications quickly and efficiently.

    ### Why Use This Chatbot?
    Whether you need quick answers, help with specific tasks, or just want to explore, our chatbot is here to assist you. It is designed to handle various types of queries and provide accurate and timely responses.

    ### Future Enhancements
    We are continually working to improve our chatbot by incorporating more advanced machine learning models, expanding the dataset, and adding new features to enhance user interaction.

    ### Contact Us
    If you have any feedback or questions about our chatbot, feel free to reach out to us. We are always looking to improve and appreciate your input.

    Thank you for using our Intent-Based Chatbot. We hope it makes your experience enjoyable and efficient!
    """)

if __name__ == '__main__':
    main()