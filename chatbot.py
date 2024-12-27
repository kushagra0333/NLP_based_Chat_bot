import json
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import csv
import datetime
import os
import re


# Load intents file
with open('intents.json') as file:
    data = json.load(file)  # data is now a list of intents, not a dictionary with 'intents'

# Initialize vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=1000)

# Preprocess the data
tags = []
patterns = []
for intent in data:  # Loop directly over the list (data is already a list of intent objects)
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Function to extract source and destination from user input
def extract_locations(input_text):
    # Regular expression to extract source and destination, e.g., "from Noida to New Delhi"
    match = re.search(r"from\s(\w+(\s\w+)*)\sto\s(\w+(\s\w+)*)", input_text, re.IGNORECASE)
    if not match:
        # If "from to" format doesn't work, try a more general pattern like "travel Noida to Delhi"
        match = re.search(r"travel\s(\w+(\s\w+)*)\sto\s(\w+(\s\w+)*)", input_text, re.IGNORECASE)
    
    if match:
        return match.group(1), match.group(3)
    return None, None


def chatbot(input_text):
    # Extract source and destination
    source, destination = extract_locations(input_text)
    
    # Predicting the tag
    input_text_transformed = vectorizer.transform([input_text])
    tag = clf.predict(input_text_transformed)[0]
    
    # Generate response
    for intent in data:  # Loop over the list of intents
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            
            # Replace placeholders with actual values
            if source and destination:
                response = response.replace('{source}', source).replace('{destination}', destination)
            elif destination:
                response = response.replace('{destination}', destination)
            
            return response
    
    return "Sorry, I didn't understand that. Can you please clarify?"

# Function to load CSS
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the external CSS
css_path = "style.css"
load_css(css_path)

# Main app logic
def main():
    st.image('image.webp', width=500)
    st.title("CUSTOMER SUPPORT")
    # Creating sidebar options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Initialize session state for user input
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ''

    # Home page
    if choice == 'Home':
        st.write('Welcome to the customer support, please enter your message here')
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['User Inputs', 'ChatBot Response', 'Timestamp'])

        # Detect user input when pressing Enter
        def on_input_change():
            if st.session_state.user_input:
                response = chatbot(st.session_state.user_input)
                st.session_state.response = response

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([st.session_state.user_input, response, timestamp])

                # Clear input after submission
                st.session_state.user_input = ''

        # Text input that triggers on change (when user presses Enter)
        st.text_input("USER:", key="user_input", on_change=on_input_change)

        # Display the chatbot's response
        if 'response' in st.session_state:
            st.text_area("ChatBot:", value=st.session_state.response, height=120, max_chars=None)

        # Show goodbye message when the user says goodbye
        if 'response' in st.session_state and st.session_state.response.lower() in ['goodbye', 'bye']:
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

This chatbot helps you with bus bookings, cancellations, payment info, and more. It uses advanced Natural Language Processing (NLP) and machine learning to understand your queries and provide accurate responses.

### Features:
- *Intent Recognition*: Understands the context of your questions like booking, cancellations, and schedules.
- *User-Friendly Interface*: Simple and easy to interact with.
- *Real-Time Assistance*: Get answers quickly.

### Technologies:
- NLP with TfidfVectorizer
- Logistic Regression for classification
- Streamlit for easy web interface deployment

### Contact Us:
Feel free to reach out for any inquiries or feedback.

    Thank you for using our Intent-Based Chatbot. We hope it makes your experience enjoyable and efficient!
    """)

if __name__ == '__main__':
    main()
