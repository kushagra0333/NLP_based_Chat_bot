# bus schedule function need work intent and location extraction
import json
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import csv
import os
import re
import datetime


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

# Function to extract source and destination from user input
def extract_locations(input_text):
    # Regular expressions to extract source and destination
    patterns = [
        r"from\s([\w\s]+)\sto\s([\w\s]+)",  # e.g., "from Noida to New Delhi"
        r"travel\s([\w\s]+)\sto\s([\w\s]+)",  # e.g., "travel Noida to Delhi"
        r"between\s([\w\s]+)\sand\s([\w\s]+)",  # e.g., "between Noida and Delhi"
        r"([\w\s]+)\sto\s([\w\s]+)\sbus\s",  # e.g., "Noida to Delhi bus"
        r"([\w\s]+)\sto\s([\w\s]+)",  # e.g., "Noida to Delhi"
        r"schedule\sfor\s([\w\s]+)\sto\s([\w\s]+)"  # e.g., "schedule for Mumbai to Delhi"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, input_text, re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()  # Return both source and destination as strings
    
    return None, None




def get_bus_schedule(source, destination):
    source = source.strip().title()
    destination = destination.strip().title()
    print(f"Looking up schedule for {source} to {destination}")  # Debug print

    schedules = {
        ("Noida", "New Delhi"): ["8:00 AM", "12:00 PM", "6:00 PM"],
        ("Noida", "Gurgaon"): ["9:00 AM", "1:00 PM", "7:00 PM"],
        ("Mumbai", "Delhi"): ["7:00 AM", "11:00 AM", "5:00 PM"],
        ("Bangalore", "Chennai"): ["6:00 AM", "2:00 PM", "10:00 PM"],
        ("Hyderabad", "Bangalore"): ["8:00 AM", "4:00 PM", "12:00 AM"],
        ("Pune", "Mumbai"): ["5:00 AM", "11:00 AM", "6:00 PM"],
        ("Kolkata", "Delhi"): ["6:30 AM", "2:30 PM", "10:30 PM"],
        ("Ahmedabad", "Mumbai"): ["6:45 AM", "2:45 PM", "10:45 PM"],
        ("Jaipur", "Delhi"): ["7:00 AM", "3:00 PM", "11:00 PM"],
        ("Lucknow", "Delhi"): ["8:30 AM", "4:30 PM", "12:30 AM"],
        ("Chandigarh", "Delhi"): ["6:15 AM", "2:15 PM", "10:15 PM"],
        ("Goa", "Mumbai"): ["7:30 AM", "3:30 PM", "11:30 PM"],
        ("Patna", "Kolkata"): ["5:15 AM", "1:15 PM", "9:15 PM"],
        ("Nagpur", "Mumbai"): ["6:00 AM", "2:00 PM", "10:00 PM"],
        ("Madurai", "Chennai"): ["5:45 AM", "1:45 PM", "9:45 PM"],
        ("Bhopal", "Delhi"): ["7:30 AM", "3:30 PM", "11:30 PM"],
        ("Varanasi", "Delhi"): ["8:00 AM", "4:00 PM", "12:00 AM"],
        ("Indore", "Mumbai"): ["6:30 AM", "2:30 PM", "10:30 PM"],
        ("Surat", "Mumbai"): ["5:30 AM", "1:30 PM", "9:30 PM"],
        ("Amritsar", "Delhi"): ["6:45 AM", "2:45 PM", "10:45 PM"],
        ("Visakhapatnam", "Hyderabad"): ["7:15 AM", "3:15 PM", "11:15 PM"],
        ("Coimbatore", "Chennai"): ["6:00 AM", "2:00 PM", "10:00 PM"],
        ("Udaipur", "Jaipur"): ["7:00 AM", "3:00 PM", "11:00 PM"],
        ("Kanpur", "Delhi"): ["8:00 AM", "4:00 PM", "12:00 AM"]
    }

    # Create the key for lookup
    key = (source, destination)
    print(f"Lookup key: {key}")  # Debug print

    schedule = schedules.get(key, [])
    print(f"Schedule found: {schedule}")  # Debug print

    return ", ".join(schedule) or "No schedule available"


# Function to extract date and number of passengers from the user input
def extract_date_and_passengers(input_text):
    # Regex patterns for extracting date and number of passengers
    date_pattern = r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b"
    passengers_pattern = r"\b(\d+)\s*(people|persons|passengers)\b"
    
    # Extract date
    date_match = re.search(date_pattern, input_text)
    raw_date = date_match.group(1) if date_match else None
    extracted_date = None
    
    if raw_date:
        try:
            # Normalize separators
            raw_date = raw_date.replace('-', '/')
            # Try parsing the date
            if raw_date.startswith(('19', '20')):  # Format: yyyy/mm/dd
                extracted_date = datetime.datetime.strptime(raw_date, "%Y/%m/%d").strftime("%d/%m/%Y")
            else:  # Format: dd/mm/yyyy or mm/dd/yyyy
                extracted_date = datetime.datetime.strptime(raw_date, "%d/%m/%Y").strftime("%d/%m/%Y")
        except ValueError:
            extracted_date = None  # Set to None if parsing fails
    
    # Extract number of passengers
    passengers_match = re.search(passengers_pattern, input_text, re.IGNORECASE)
    passengers = int(passengers_match.group(1)) if passengers_match else None

    return extracted_date, passengers



# Mock function to get distance between two locations (in km)
def get_distance(source, destination):
    # This is a mock function. Replace it with an actual implementation if needed.
    distances = {
        ("Noida", "New Delhi"): 25,
        ("Noida", "Gurgaon"): 35,
        ("Noida", "Ghaziabad"): 15,
        ('noida', 'delhi'): 20,
        ('mumbai', 'delhi'): 1400,
        ('bangalore', 'mumbai'): 980,
        ('kolkata', 'delhi'): 1500,
        ('chennai', 'bangalore'): 350,
        ('pune', 'mumbai'): 150,
        ('hyderabad', 'bangalore'): 570,
        ('jaipur', 'delhi'): 280,
        ('lucknow', 'delhi'): 550,
        ('ahmedabad', 'mumbai'): 530,
        ('bhopal', 'delhi'): 770,
        ('kochi', 'chennai'): 700,
        ('nagpur', 'mumbai'): 820,
        ('patna', 'kolkata'): 580,
        ('chandigarh', 'delhi'): 240,
        ('goa', 'mumbai'): 590,
        ('kanpur', 'delhi'): 480,
        ('surat', 'mumbai'): 290,
        ('indore', 'mumbai'): 585,
        ('agra', 'delhi'): 210,
        ('varanasi', 'delhi'): 835,
        ('amritsar', 'delhi'): 450,
        ('jodhpur', 'jaipur'): 340,
        ('coimbatore', 'chennai'): 510,
        ('visakhapatnam', 'hyderabad'): 620,
        ('udaipur', 'jaipur'): 395,
        ('guwahati', 'kolkata'): 990,
        ('madurai', 'chennai'): 460
        # Add more mock distances as needed
    }
    return distances.get((source, destination), random.randint(10, 50))

# Function to handle chatbot responses
def chatbot(input_text):
    # Extract source and destination for schedule-related queries
    source, destination = extract_locations(input_text)
    print(f"Source: {source}, Destination: {destination}")  # Debug print
    # Extract date and passengers if necessary
    date, passengers = extract_date_and_passengers(input_text)
    
    # Predict the intent (e.g., schedule, fare)
    input_text_transformed = vectorizer.transform([input_text])
    tag = clf.predict(input_text_transformed)[0]

    # Debugging info
    print(f"User input: {input_text}, Intent tag: {tag}")

    # General intent-based response
    for intent in data:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            
            if source and destination:
                response = response.replace("{source}", source).replace("{destination}", destination)
                # Add the schedule or fare based on intent
                if tag == "bus_schedule":
                    schedule = get_bus_schedule(source, destination)
                    response = response.replace("{schedule}", schedule if schedule else "No schedule available")
                elif tag == "fare_inquiry":
                    distance = get_distance(source, destination)
                    fare = distance * 9  # Rs 9 per km
                    response = response.replace("{fare}", str(fare))

            if date and passengers:
                response = response.replace("{date}", date).replace("{number}", str(passengers))

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
    if 'response' not in st.session_state:
        st.session_state.response = ''

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
        if st.session_state.response:
            st.text_area("ChatBot:", value=st.session_state.response, height=120, max_chars=None)

        # Show goodbye message when the user says goodbye
        if st.session_state.response.lower() in ['goodbye', 'bye']:
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