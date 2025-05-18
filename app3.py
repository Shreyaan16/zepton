import os
import pandas as pd
import numpy as np
import statistics
import re
from datetime import datetime
import google.generativeai as genai
import streamlit.components.v1 as components
import streamlit as st
import time
from io import StringIO

# Initialize API key from environment variables
# This will work with both local .env files and deployment secrets
GEMINI_API_KEY = None

# For local development: Try to load from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# First check for a Streamlit secret
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, AttributeError):
    # If not in Streamlit secrets, try environment variable
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# If API key exists, configure the API
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        st.write("API Key loaded successfully!")
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")
        GEMINI_API_KEY = None
else:
    st.warning("Gemini API key not found. Some features will be limited.")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'df' not in st.session_state:
    st.session_state.df = None

if 'summary' not in st.session_state:
    st.session_state.summary = None

# st.markdown("""
# <div id="video-overlay">
#     <video id="intro-video" autoplay muted>
#         <source src="https://cdn.pixabay.com/video/2023/01/09/145864-787701151_large.mp4" type="video/mp4">
#         Your browser does not support the video tag.
#     </video>
# </div>

# <style>
# #video-overlay {
#     position: fixed;
#     top: 0;
#     left: 0;
#     width: 100%;
#     height: 100%;
#     background: #000;
#     z-index: 9999;
#     display: flex;
#     justify-content: center;
#     align-items: center;
# }

# #intro-video {
#     width: 100%;
#     height: 100%;
#     object-fit: cover;
# }

# .stApp {
#     visibility: hidden;
# }

# .stApp.visible {
#     visibility: visible;
# }
# </style>

# <script>
# document.addEventListener('DOMContentLoaded', function() {
#     const video = document.getElementById('intro-video');
#     const overlay = document.getElementById('video-overlay');
#     const app = document.querySelector('.stApp');

#     video.onended = function() {
#         overlay.style.display = 'none';
#         app.classList.add('visible');
#     };

#     // Fallback: If video fails to load or play, show content after 5 seconds
#     setTimeout(function() {
#         overlay.style.display = 'none';
#         app.classList.add('visible');
#     }, 5000);
# });
# </script>
# """, unsafe_allow_html=True)

# Function to check if dataframe has all required columns
def validate_df_columns(df):
    required_columns = ['NPI', 'State', 'Region', 'Speciality', 
                       'Usage Time (mins)', 'Count of Survey Attempts',
                       'Login Date', 'Login Time', 'Logout Date', 'Logout Time']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, missing_columns
    return True, []

# Function to process uploaded CSV data
def process_uploaded_data(df):
    try:
        # Check if required columns exist
        is_valid, missing_columns = validate_df_columns(df)
        if not is_valid:
            return None, f"CSV file is missing required columns: {', '.join(missing_columns)}"
        
        # Convert date and time columns to datetime objects
        df['Login DateTime'] = pd.to_datetime(df['Login Date'] + ' ' + df['Login Time'])
        df['Logout DateTime'] = pd.to_datetime(df['Logout Date'] + ' ' + df['Logout Time'])
        
        # Verify that Usage Time is accurate
        calculated_usage = (df['Logout DateTime'] - df['Login DateTime']).dt.total_seconds() / 60
        # If there's a significant difference, update the Usage Time
        if np.abs((calculated_usage - df['Usage Time (mins)']).mean()) > 1:
           df['Usage Time (mins)'] = calculated_usage
    
        return df, None
    except Exception as e:
        return None, f"Error processing data: {str(e)}"

# Function to generate data summary
def generate_data_summary(df):
    df['Login Hour'] = df['Login DateTime'].dt.hour
    # Find the most frequent login hour
    frequent_login_hour = df['Login Hour'].mode()[0]
    summary = {
        "total_records": len(df),
        "date_range": {
            "min": df['Login Date'].min(),
            "max": df['Login Date'].max()
        },
        "specialties": df['Speciality'].unique().tolist(),
        "regions": df['Region'].unique().tolist(),
        "states": df['State'].unique().tolist(),
        "usage_time": {
            "mean": df['Usage Time (mins)'].mean(),
            "median": df['Usage Time (mins)'].median(),
            "min": df['Usage Time (mins)'].min(),
            "max": df['Usage Time (mins)'].max()
        },
        "survey_attempts": {
            "mean": df['Count of Survey Attempts'].mean(),
            "median": df['Count of Survey Attempts'].median(),
            "min": df['Count of Survey Attempts'].min(),
            "max": df['Count of Survey Attempts'].max()
        },
        "frequent_login_hour": frequent_login_hour
    }
    return summary

# Function to perform statistical analysis
def analyze_data(df, column, operation, filter_dict=None):
    # Apply filters if any
    if filter_dict:
        for key, value in filter_dict.items():
            if key in df.columns:
                df = df[df[key] == value]
    
    # Perform the statistical operation
    if operation == 'mean':
        return df[column].mean()
    elif operation == 'median':
        return df[column].median()
    elif operation == 'mode':
        try:
            return statistics.mode(df[column])
        except:
            return "No unique mode found"
    elif operation == 'min':
        return df[column].min()
    elif operation == 'max':
        return df[column].max()
    elif operation == 'count':
        return len(df)
    elif operation == 'sum':
        return df[column].sum()
    elif operation == 'std':
        return df[column].std()
    elif operation == 'unique':
        return df[column].unique().tolist()
    else:
        return f"Operation {operation} not supported"

# Function to get details for a specific NPI
def get_npi_details(df, npi, attribute=None):
    try:
        npi = int(npi)  # Ensure NPI is an integer
        record = df[df['NPI'] == npi]
        if record.empty:
            return f"No record found for NPI {npi}"
        
        record = record.iloc[0]
        if attribute:
            # Map user-friendly attribute names to DataFrame column names
            attribute_map = {
                'state': 'State',
                'region': 'Region',
                'speciality': 'Speciality',
                'specialty': 'Speciality',
                'usage time': 'Usage Time (mins)',
                'survey attempts': 'Count of Survey Attempts',
                'login datetime': 'Login DateTime',
                'logout datetime': 'Logout DateTime'
            }
            attribute = attribute.lower()
            if attribute in attribute_map:
                column = attribute_map[attribute]
                value = record[column]
                if column == 'Login DateTime':
                    value = f"{record['Login Date']} {record['Login Time']}"
                elif column == 'Logout DateTime':
                    value = f"{record['Logout Date']} {record['Logout Time']}"
                return f"{attribute.capitalize()} for NPI {npi}: {value}"
            return f"Attribute '{attribute}' not recognized"
        
        # Return all details if no specific attribute is requested
        details = f"""
        NPI: {record['NPI']}
        State: {record['State']}
        Region: {record['Region']}
        Speciality: {record['Speciality']}
        Usage Time (mins): {record['Usage Time (mins)']}
        Count of Survey Attempts: {record['Count of Survey Attempts']}
        Login DateTime: {record['Login Date']} {record['Login Time']}
        Logout DateTime: {record['Logout Date']} {record['Logout Time']}
        """
        return details.strip()
    except Exception as e:
        return f"Error retrieving details for NPI {npi}: {e}"

# Function to list available models (for debugging)
def list_available_models():
    try:
        models = genai.list_models()
        return [model.name for model in models if 'generateContent' in model.supported_generation_methods]
    except Exception as e:
        return f"Error listing models: {e}"

# Function to get synonyms for column names
def get_column_synonyms(column_name):
    synonyms = {
        'Usage Time (mins)': ['time', 'usage', 'duration', 'minutes', 'login time', 'logout time'],
        'Count of Survey Attempts': ['survey', 'attempts', 'surveys', 'tries'],
        'Speciality': ['specialty', 'specialization', 'field', 'specialities', 'specialties'],
        'Region': ['area', 'zone', 'location', 'regions'],
        'State': ['state', 'province', 'territory', 'states'],
        'NPI': ['npi', 'provider', 'providers', 'professional', 'professionals']
    }
    return synonyms.get(column_name, [])

# Function to filter dataframe based on question
def filter_dataframe_from_question(df, question_lower):
    filters = {}
    
    # Check for region filters
    regions = df['Region'].unique()
    for region in regions:
        if region.lower() in question_lower:
            filters['Region'] = region
    
    # Check for state filters
    states = df['State'].unique()
    for state in states:
        if state.lower() in question_lower:
            filters['State'] = state
    
    # Check for specialty filters
    specialties = df['Speciality'].unique()
    for specialty in specialties:
        if specialty.lower() in question_lower:
            filters['Speciality'] = specialty
    
    if filters:
        filtered_df = df.copy()
        for key, value in filters.items():
            filtered_df = filtered_df[filtered_df[key] == value]
        return filtered_df
    
    return df

# Function to process questions with Gemini API
def process_question(question, df, summary, chat_history):
    question_lower = question.lower()
    
    # Build context from chat history for continuity
    conversation_context = ""
    if chat_history:
        recent_history = chat_history[-5:]
        for i, exchange in enumerate(recent_history):
            conversation_context += f"Q{len(chat_history)-len(recent_history)+i+1}: {exchange['question']}\n"
            conversation_context += f"A{len(chat_history)-len(recent_history)+i+1}: {exchange['answer']}\n\n"
    
    # Handle specialties question locally
    if 'specialities' in question_lower or 'specialties' in question_lower:
        specialties = summary['specialties']
        return f"Here's the list of specialties in the dataset: {', '.join(specialties)}."
    
    # Check for NPI range questions
    if 'npi range' in question_lower or 'npi ranges' in question_lower:
        npi_min = df['NPI'].min()
        npi_max = df['NPI'].max()
        return f"The NPIs in the dataset range from {npi_min} to {npi_max}."
    
    if 'frequent login time' in question_lower:
     df['Login Hour'] = df['Login DateTime'].dt.hour
     frequent_hour = df['Login Hour'].mode()[0]
     return f"The most frequent login hour is {frequent_hour}:00."
    
    if 'total count of npi' in question_lower or 'how many npi' in question_lower:
     return f"The total number of NPIs in the dataset is {summary['total_records']}."

    # Check for specific NPI attribute questions
    npi_match = re.search(r'\b\d{10}\b', question)
    if npi_match:
        npi = npi_match.group(0)
        attributes = ['state', 'region', 'speciality', 'specialty', 'usage time', 
                      'survey attempts', 'login datetime', 'logout datetime']
        for attr in attributes:
            if attr in question_lower:
                return get_npi_details(df, npi, attribute=attr)
        return f"Here's everything I have for NPI {npi}:\n\n{get_npi_details(df, npi)}"
    
    # Check for questions about most NPIs by category (e.g., state, region, specialty)
    if any(keyword in question_lower for keyword in ['most', 'highest', 'top']) and \
       any(npi_syn in question_lower for npi_syn in get_column_synonyms('NPI')):
        category_map = {
            'State': ['state', 'states', 'province', 'territory'],
            'Region': ['region', 'area', 'zone', 'location', 'regions'],
            'Speciality': ['specialty', 'specialities', 'specialization', 'field', 'specialties']
        }
        
        target_category = None
        for category, synonyms in category_map.items():
            if any(synonym in question_lower for synonym in synonyms):
                target_category = category
                break
        
        if target_category and target_category in df.columns:
            filtered_df = filter_dataframe_from_question(df, question_lower)
            if filtered_df.empty:
                return "Looks like there’s no data matching your filters. Try tweaking your question!"
            
            counts = filtered_df[target_category].value_counts()
            if counts.empty:
                return f"No {target_category.lower()} found in the filtered data."
            
            max_count = counts.max()
            top_categories = counts[counts == max_count].index.tolist()
            
            if len(top_categories) == 1:
                return f"The {target_category.lower()} with the most NPIs is {top_categories[0]} with {max_count} providers."
            else:
                return f"Multiple {target_category.lower()}s are tied for the most NPIs ({max_count} providers each): {', '.join(top_categories)}."
    
    # Check for statistical questions related to columns and NPIs
    stat_keywords = ['max', 'maximum', 'highest', 'min', 'minimum', 'lowest']
    if any(keyword in question_lower for keyword in stat_keywords):
        operation = 'max' if any(k in question_lower for k in ['max', 'maximum', 'highest']) else 'min'
        column_map = {
            'Usage Time (mins)': ['usage time', 'time', 'login time', 'logout time', 'duration', 'minutes'],
            'Count of Survey Attempts': ['survey attempts', 'survey', 'attempts', 'tries']
        }
        
        target_column = None
        for column, synonyms in column_map.items():
            if any(synonym in question_lower for synonym in synonyms) or column.lower() in question_lower:
                target_column = column
                break
        
        if target_column and target_column in df.columns:
            filtered_df = filter_dataframe_from_question(df, question_lower)
            if filtered_df.empty:
                return "I couldn’t find any data matching your filters. Maybe try a different region or specialty?"
            
            stat_value = filtered_df[target_column].max() if operation == 'max' else filtered_df[target_column].min()
            npi_list = filtered_df[filtered_df[target_column] == stat_value]['NPI'].tolist()
            
            if not npi_list:
                return f"No NPIs found for the {operation} {target_column.lower()}."
            
            unit = " minutes" if target_column == 'Usage Time (mins)' else ""
            response = f"The {operation} {target_column.lower()} is {stat_value:.2f}{unit}, and here’s who had it:\n\n"
            for npi in npi_list:
                details = get_npi_details(df, npi)
                response += f"{details}\n\n"
            return response.strip()
    
    # Check for general NPI questions without a specific number
    if 'npi' in question_lower and not any(digit in question for digit in '0123456789'):
        if any(keyword in question_lower for keyword in ['average', 'mean', 'median', 'total', 'sum']):
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in question_lower or any(synonym in question_lower for synonym in get_column_synonyms(col)):
                    filtered_df = filter_dataframe_from_question(df, question_lower)
                    if isinstance(filtered_df, pd.DataFrame):
                        try:
                            if 'usage time' in question_lower or 'time' in question_lower:
                                result = filtered_df['Usage Time (mins)'].mean()
                                return f"On average, usage time is {result:.2f} minutes across the filtered data."
                            elif 'survey' in question_lower or 'attempts' in question_lower:
                                result = filtered_df['Count of Survey Attempts'].mean()
                                return f"The average number of survey attempts is {result:.2f}."
                        except Exception as e:
                            pass
        return "Could you share a specific 10-digit NPI or clarify what stats you’re looking for?"
    
    # Check for references to previous answers
    if any(ref in question_lower for ref in ['previous answer', 'last question', 'you just said', 'earlier response', 'previous response']):
        if not chat_history:
            return "I don’t have any previous chats to refer to yet. What’s your question?"
        
        conversation_context = "Here’s what we talked about recently:\n\n"
        for i, exchange in enumerate(chat_history[-3:]):
            conversation_context += f"Q{len(chat_history)-3+i+1}: {exchange['question']}\n"
            conversation_context += f"A{len(chat_history)-3+i+1}: {exchange['answer']}\n\n"
    
    # Check for common statistical questions
    if 'average' in question_lower or 'mean' in question_lower:
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in question_lower or any(synonym in question_lower for synonym in get_column_synonyms(col)):
                filtered_df = filter_dataframe_from_question(df, question_lower)
                if isinstance(filtered_df, pd.DataFrame):
                    try:
                        if 'usage time' in question_lower or 'time' in question_lower:
                            result = filtered_df['Usage Time (mins)'].mean()
                            return f"On average, usage time is {result:.2f} minutes."
                        elif 'survey' in question_lower or 'attempts' in question_lower:
                            result = filtered_df['Count of Survey Attempts'].mean()
                            return f"The average number of survey attempts is {result:.2f}."
                    except Exception as e:
                        pass
    
    # Use Gemini API if available, otherwise use local fallback
    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            context = f"""
            You’re a friendly data analyst helping with medical professional data. The dataset has:
            - NPI: Unique ID for medical professionals
            - State: US state code
            - Usage Time (mins): Time spent using the system
            - Region: Geographic region (Northeast, Midwest, etc.)
            - Speciality: Medical specialty
            - Count of Survey Attempts: Number of survey attempts
            - Login Date/Time, Logout Date/Time
            
            Dataset summary:
            - Total records: {summary['total_records']}
            - Date range: {summary['date_range']['min']} to {summary['date_range']['max']}
            - Most frequent login hour: {summary['frequent_login_hour']}:00
            - Specialties: {', '.join(summary['specialties'][:5]) if len(summary['specialties']) > 5 else ', '.join(summary['specialties'])}... {f"(and {len(summary['specialties'])-5} more)" if len(summary['specialties']) > 5 else ""}
            - Regions: {', '.join(summary['regions'])}
            - States: {', '.join(summary['states'][:10]) if len(summary['states']) > 10 else ', '.join(summary['states'])}... {f"(and {len(summary['states'])-10} more)" if len(summary['states']) > 10 else ""}
            - Usage Time: Avg = {summary['usage_time']['mean']:.2f} mins, Median = {summary['usage_time']['median']:.2f}, Min = {summary['usage_time']['min']}, Max = {summary['usage_time']['max']}
            - Survey Attempts: Avg = {summary['survey_attempts']['mean']:.2f}, Median = {summary['survey_attempts']['median']:.2f}, Min = {summary['survey_attempts']['min']}, Max = {summary['survey_attempts']['max']}
            
            {conversation_context}
            
            Current question: {question}
            
            Answer in a friendly, conversational way based on the data. Be precise with stats and include numbers. If more analysis is needed, let me know. Keep the context of previous questions in mind.
            """
            
            response = model.generate_content([context])
            response_text = response.text
            
            if "specific analysis" in response_text.lower() or "more data" in response_text.lower():
                top_specialties = df['Speciality'].value_counts().head(5).to_dict()
                region_usage = df.groupby('Region')['Usage Time (mins)'].mean().to_dict()
                
                additional_info = f"""
                Here’s some extra insight:
                - Top specialties by count: {top_specialties}
                - Average usage time by region: {region_usage}
                """
                response_text += "\n\n" + additional_info
            
            return response_text
        except Exception as e:
            return local_fallback_processing(question, df, summary, chat_history)
    else:
        return local_fallback_processing(question, df, summary, chat_history)

# Function to handle local processing when API is not accessible
def local_fallback_processing(question, df, summary, chat_history):
    question_lower = question.lower()
    
    # Basic question answering logic
    if 'how many' in question_lower:
        if 'specialties' in question_lower or 'specialities' in question_lower:
            return f"There are {len(summary['specialties'])} specialties in the dataset."
        elif 'regions' in question_lower:
            return f"There are {len(summary['regions'])} regions in the dataset."
        elif 'states' in question_lower:
            return f"There are {len(summary['states'])} states in the dataset."
        elif 'records' in question_lower or 'entries' in question_lower or 'rows' in question_lower:
            return f"The dataset has {summary['total_records']} records."
    

    if 'frequent login time' in question_lower:
     df['Login Hour'] = df['Login DateTime'].dt.hour
     frequent_hour = df['Login Hour'].mode()[0]
     return f"The most frequent login hour is {frequent_hour}:00."
    if 'total count of npi' in question_lower or 'how many npi' in question_lower:
     return f"The total number of NPIs in the dataset is {summary['total_records']}."

    # Statistical questions
    if 'average' in question_lower or 'mean' in question_lower:
        if 'usage time' in question_lower:
            return f"On average, providers spend {summary['usage_time']['mean']:.2f} minutes using the system."
        elif 'survey attempts' in question_lower:
            return f"The average number of survey attempts is {summary['survey_attempts']['mean']:.2f}."
    
    if 'median' in question_lower:
        if 'usage time' in question_lower:
            return f"The median usage time is {summary['usage_time']['median']:.2f} minutes."
        elif 'survey attempts' in question_lower:
            return f"The median number of survey attempts is {summary['survey_attempts']['median']:.2f}."
    
    if 'minimum' in question_lower or 'min' in question_lower:
        if 'usage time' in question_lower:
            return f"The shortest usage time is {summary['usage_time']['min']} minutes."
        elif 'survey attempts' in question_lower:
            return f"The fewest survey attempts is {summary['survey_attempts']['min']}."
    
    if 'maximum' in question_lower or 'max' in question_lower:
        if 'usage time' in question_lower:
            return f"The longest usage time is {summary['usage_time']['max']} minutes."
        elif 'survey attempts' in question_lower:
            return f"The most survey attempts is {summary['survey_attempts']['max']}."
    
    # Top entries
    if 'top' in question_lower:
        if 'specialties' in question_lower or 'specialities' in question_lower:
            top_n = 5
            for num in re.findall(r'\d+', question):
                top_n = int(num)
                break
            
            top_specs = df['Speciality'].value_counts().head(top_n)
            result = "Here are the top specialties by number of providers:\n"
            for specialty, count in top_specs.items():
                result += f"- {specialty}: {count} providers\n"
            return result
        
        if 'regions' in question_lower:
            top_regions = df['Region'].value_counts().head(5)
            result = "Here are the top regions by number of providers:\n"
            for region, count in top_regions.items():
                result += f"- {region}: {count} providers\n"
            return result
    
    return "I’m not quite sure what you’re asking. Could you rephrase or provide more details?"

# Function to display chat message with avatar
def display_message(is_user, message):
    if is_user:
        message_container = st.chat_message("user")
        with message_container:
            st.markdown(message)
    else:
        message_container = st.chat_message("assistant")
        with message_container:
            st.markdown(message)

# Main Streamlit UI
def main():
    st.set_page_config(
        page_title="Medical Data Analyst",
        page_icon="🩺",
        layout="wide"
    )
    
    st.markdown("""
<style>

/* Global font family */
.stApp, .st-emotion-cache-br351g p, h1, h2, h3, h4 {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
}
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;700;800&display=swap');

html, body, .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background: linear-gradient(135deg, #ff6f61, #ffb347, #ffcc00);
    background-size: 300% 300%;
    animation: gradientBG 20s ease infinite;
    color: white;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Floating icons */
.floating {
    position: absolute;
    font-size: 40px;
    animation: floatUp 14s linear infinite;
    opacity: 0.8;
    pointer-events: none;
    z-index: 1;
}
.floating:nth-child(1) { left: 10%; animation-delay: 0s; }
.floating:nth-child(2) { left: 30%; animation-delay: 3s; }
.floating:nth-child(3) { left: 50%; animation-delay: 6s; }
.floating:nth-child(4) { left: 70%; animation-delay: 2s; }
.floating:nth-child(5) { left: 85%; animation-delay: 5s; }

@keyframes floatUp {
    0% { bottom: -10%; transform: translateY(0); opacity: 0; }
    50% { opacity: 1; }
    100% { bottom: 110%; transform: translateY(-100vh); opacity: 0; }
}

/* Headings */
h1, h2, h3, h4 {
    text-align: center;
    font-weight: 800;
    font-size: 3rem;
    color: #1a1a1a;
    text-shadow: 2px 2px 6px rgba(255,255,255,0.4);
    margin-bottom: 1rem;
}

/* Tabs */
.stTabs {
    display: flex;
    justify-content: center;
    margin-top: 1rem;
}
.stTabs button {
    font-size: 1.8rem !important;
    font-weight: 800;
    color: #222222 !important;
    background-color: transparent !important;
    border: none;
    margin: 0 15px;
}

/* File uploader */
.stFileUploader {
    border: 2px dashed rgba(255,255,255,0.6);
    background-color: rgba(255,255,255,0.1);
    backdrop-filter: blur(6px);
    border-radius: 16px;
    color: white;
    transition: 0.3s ease;
}
.stFileUploader:hover {
    border-color: #ffcc00;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(to right, #222222, #444444);
    border: none;
    color: white;
    font-weight: 700;
    font-size: 1.2rem;
    padding: 12px 28px;
    border-radius: 30px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(to right, #111111, #333333);
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

/* Floating emojis container */
.floating-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 0;
    overflow: hidden;
}
                /*my styels*/
                .st-emotion-cache-10c9vv9 {
                color- white !important;
                font-size: 22px !important;
                font-weight: 600 !important;
                }
                .st-emotion-cache-iyz50i:hover {
    border-color: 2px solid black !important;
                -webkit-text-fill-color: white !important;
    background-color: white !important;
                }
                .stButton > button {
                width:250px !important;
                height: 100px !important;
                color:white !important;
                }
                .stFileUploader {
                width:650px !important;
                height: 200px !important;
                border:linear-gradient(to right, #ff6ec4, #7873f5) !important;
                background-color: rgba(120, 115, 245, 0.2) !important;
                }
                .st-emotion-cache-br351g {
                font-size:28px !important;
                font-weight: 600 !important;
                   background: linear-gradient(90deg, #1B263B, #415A77);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;



                }
 .st-emotion-cache-1s2v671 {
                width:550px !important;
                }
                .st-emotion-cache-1gulkj5 {
                width:550px !important;
                }





                div[data-testid="stForm"] {
    background: #ffffff !important; /* White background for contrast */
    border: 3px solid transparent !important; /* Reserve space for gradient */
    border-image: linear-gradient(45deg, #1B263B, #415A77) 1 !important; /* Bluish gradient border */
    border-image-slice: 1 !important;
    border-radius: 12px !important; /* Rounded corners */
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important; /* Subtle shadow */
    padding: 20px !important; /* Add padding inside the form */
    max-width: 600px !important; /* Constrain width for better appearance */
    margin: 0 auto !important; /* Center the form */
}

/* Style the label ("Ask a Question") */
div[data-testid="stForm"] label[data-testid="stWidgetLabel"] p {
    font-size: 20px !important; /* Increase font size */
    font-weight: 600 !important; /* Make it bold */
    color: #1B263B !important; /* Dark navy color for text */
    margin-bottom: 10px !important; /* Add spacing below */
}

/* Style the textarea */
div[data-testid="stForm"] div[data-testid="stTextAreaRootElement"] textarea {
    font-size: 18px !important; /* Increase font size for placeholder and input text */
    color: #000000 !important; /* Black text for better visibility */
    background: #ffffff !important; /* White background */
    border: 2px solid #1B263B !important; /* More visible border */
    border-radius: 8px !important;
    padding: 12px !important; /* More padding for comfort */
    height: 120px !important; /* Increase height for better usability */
    caret-color: #1B263B !important; /* Make cursor more visible */
}

/* Style the placeholder text */
div[data-testid="stForm"] div[data-testid="stTextAreaRootElement"] textarea::placeholder {
    font-size: 18px !important; /* Increase placeholder font size */
    color: #888 !important; /* Lighter gray for placeholder */
}

/* Style the buttons ("Send" and "Clear initializes") */
div[data-testid="stForm"] button[data-testid="stBaseButton-secondaryFormSubmit"] {
    font-size: 16px !important; /* Increase font size */
    font-weight: 500 !important; /* Medium weight */
    background: linear-gradient(90deg, #1B263B, #415A77) !important; /* Gradient background for buttons */
    color: white !important; /* White text */
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important; /* More padding for larger buttons */
    transition: transform 0.2s ease !important; /* Add hover effect */
    margin: 0 auto !important; /* Center align buttons */
    display: block !important; /* Ensure proper centering */
}

/* Center the columns containing buttons */
div[data-testid="stForm"] [data-testid="column"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

/* Add hover effect for buttons */
div[data-testid="stForm"] button[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
    transform: scale(1.05) !important; /* Slight scale on hover */
}

.st-emotion-cache-16tyu1 p, .st-emotion-cache-16tyu1 ol, .st-emotion-cache-16tyu1 ul, .st-emotion-cache-16tyu1 dl, .st-emotion-cache-16tyu1 li {
                font-size:24px !important;
                font-weight: 600 !important;
                }
 .st-emotion-cache-1fmytai {
                background:linear-gradient(to right, #1e3c72, #2a5298, #6a0572);
                font-size:40px !important;
                font-weight: 600 !important;}
                
h1, h2{
    background: linear-gradient(90deg, #8e2de2, #4a00e0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

</style>       
""", unsafe_allow_html=True)

    st.markdown("""
<div class="floating-container">
    <div class="floating">🤖</div>
    <div class="floating">✨</div>
    <div class="floating">⚡</div>
    <div class="floating">🧠</div>
    <div class="floating">💫</div>
</div>
""", unsafe_allow_html=True)
    # Header
    st.markdown("""
    <div class="main-header-box">
                <span style="font-size: 32px;">🩺</span>
    <h1>Medical Data Analysis Assistant</h1>
</div>
    """, unsafe_allow_html=True)
    #st.markdown("Explore medical professional data with ease. Upload your dataset and ask questions to uncover insights.", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2 = st.tabs(["📊 Data Management", "💬 Chat Interface"])

    # Tab 1: Data Management
    with tab1:
        st.markdown("### Manage Your Data")
        # Data Upload Section
        with st.container():
            st.markdown("#### Upload Dataset", unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                uploaded_file = st.file_uploader("Drag and drop or click to upload a CSV file", type=['csv'])
                if uploaded_file is not None:
                    with st.spinner("Processing your data..."):
                        try:
                            df = pd.read_csv(uploaded_file)
                            processed_df, error_msg = process_uploaded_data(df)
                            
                            if processed_df is not None:
                                st.session_state.df = processed_df
                                st.session_state.summary = generate_data_summary(processed_df)
                                st.success(f"Dataset loaded! Found {len(processed_df)} records.")
                            else:
                                st.error(error_msg)
                        except Exception as e:
                            st.error(f"Oops, something went wrong: {e}")
                
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("Load Sample Data", use_container_width=True):
                        with st.spinner("Loading sample data..."):
                            try:
                                sample_data = {
                                    'NPI': [1234567890, 2345678901, 3456789012, 4567890123, 5678901234],
                                    'State': ['CA', 'NY', 'TX', 'FL', 'IL'],
                                    'Region': ['West', 'Northeast', 'South', 'South', 'Midwest'],
                                    'Speciality': ['Cardiology', 'Neurology', 'Pediatrics', 'Oncology', 'Family Medicine'],
                                    'Usage Time (mins)': [45, 60, 30, 75, 25],
                                    'Count of Survey Attempts': [3, 5, 2, 4, 1],
                                    'Login Date': ['2025-01-15', '2025-01-16', '2025-01-17', '2025-01-18', '2025-01-19'],
                                    'Login Time': ['09:00:00', '10:30:00', '08:15:00', '13:45:00', '11:20:00'],
                                    'Logout Date': ['2025-01-15', '2025-01-16', '2025-01-17', '2025-01-18', '2025-01-19'],
                                    'Logout Time': ['09:45:00', '11:30:00', '08:45:00', '15:00:00', '11:45:00']
                                }
                                df = pd.DataFrame(sample_data)
                                
                                processed_df, error_msg = process_uploaded_data(df)
                                if processed_df is not None:
                                    st.session_state.df = processed_df
                                    st.session_state.summary = generate_data_summary(processed_df)
                                    st.success(f"Sample data loaded! Found {len(processed_df)} records.")
                                else:
                                    st.error(error_msg)
                            except Exception as e:
                                st.error(f"Error loading sample data: {e}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.df is not None:
            # Quick NPI Lookup
            st.markdown("#### Quick NPI Lookup", unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                col1, col2 = st.columns([3, 1])
                with col1:
                    npi_input = st.text_input("Enter a 10-digit NPI", key="npi_lookup")
                with col2:
                    if st.button("Search", use_container_width=True):
                        if npi_input and npi_input.isdigit() and len(npi_input) == 10:
                            result = get_npi_details(st.session_state.df, npi_input)
                            st.code(result, language="text")
                        else:
                            st.error("Please enter a valid 10-digit NPI.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Data Statistics
            st.markdown("#### Data Insights", unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                summary = st.session_state.summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", summary['total_records'])
                with col2:
                    st.metric("Avg Usage Time", f"{summary['usage_time']['mean']:.1f} mins")
                with col3:
                    st.metric("Avg Survey Attempts", f"{summary['survey_attempts']['mean']:.1f}")
                
                st.markdown(f"**Date Range**: {summary['date_range']['min']} to {summary['date_range']['max']}")
                
                st.markdown("##### Analyze Data", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    stat_column = st.selectbox("Select Column", 
                                              ['Usage Time (mins)', 'Count of Survey Attempts'],
                                              key="stat_column")
                with col2:
                    stat_operation = st.selectbox("Select Operation", 
                                                 ['mean', 'median', 'min', 'max', 'std'],
                                                 key="stat_op")
                
                filter_option = st.selectbox("Filter By", 
                                            ['None', 'Region', 'Speciality', 'State'],
                                            key="filter_option")
                
                if filter_option != 'None':
                    filter_value = st.selectbox(f"Select {filter_option}", 
                                               st.session_state.df[filter_option].unique(),
                                               key="filter_value")
                    filter_dict = {filter_option: filter_value}
                else:
                    filter_dict = None
                
                if st.button("Calculate", use_container_width=True):
                    result = analyze_data(st.session_state.df, stat_column, stat_operation, filter_dict)
                    if isinstance(result, (float, int)):
                        st.metric("Result", f"{result:.2f}")
                    else:
                        st.write("Result:", result)
                st.markdown('</div>', unsafe_allow_html=True)

    # Tab 2: Chat Interface
    with tab2:
        st.markdown('<div class="chat-section">', unsafe_allow_html=True)
        st.markdown("### Chat with Your Data")
        if st.session_state.df is None:
          st.info("Please upload a dataset in the Data Management tab to start chatting.")
          st.markdown('</div>', unsafe_allow_html=True)
          return
        st.markdown('</div>', unsafe_allow_html=True)

        # Chat history
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        chat_container = st.container()
        with chat_container:
         for message in st.session_state.chat_history:
            display_message(True, message["question"])
            display_message(False, message["answer"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Input form
        with st.container():
         st.markdown('<div class="card">', unsafe_allow_html=True)
         with st.form(key="question_form", clear_on_submit=True):
            user_question = st.text_area(
                "Ask a Question",
                placeholder="E.g., Which state has the most NPIs? or What’s the NPI with max usage time?",
                key="question_input",
                height=100
            )
            col1, col2 = st.columns([1, 1])
            with col1:
                submit_button = st.form_submit_button("Send", use_container_width=True)
            with col2:
                if st.form_submit_button("Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
         if submit_button and user_question:
            display_message(True, user_question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = process_question(
                        user_question, 
                        st.session_state.df, 
                        st.session_state.summary,
                        st.session_state.chat_history
                    )
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": response,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.markdown(response)
            st.rerun()
         st.markdown('</div>', unsafe_allow_html=True)
 
        # Conversation History
        st.markdown('<div class="conversation-history-section">', unsafe_allow_html=True)
        st.markdown("#### Conversation History")
        if st.session_state.chat_history:
         for i, exchange in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {exchange['question'][:50]}..." if len(exchange['question']) > 50 else f"Q{i+1}: {exchange['question']}"):
                st.markdown(f"**Question**: {exchange['question']}")
                st.markdown(f"**Answer**: {exchange['answer']}")
        else:
          st.write("No questions asked yet. Start chatting above!")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sample Questions
        if not st.session_state.chat_history:
            st.markdown("#### Try These Questions", unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                sample_questions = [
                    ("What are all the specialties?", "sample1"),
                    ("What is the average usage time?", "sample2"),
                    ("Which region has the highest survey attempts?", "sample3"),
                    ("Show me the NPI range in the dataset", "sample4")
                ]
                for idx, (col, (question, key)) in enumerate(zip([col1, col2, col3, col4], sample_questions)):
                    with col:
                        if st.button(question, key=key, use_container_width=True):
                            st.session_state.chat_history.append({
                                "question": question,
                                "answer": process_question(
                                    question, 
                                    st.session_state.df, 
                                    st.session_state.summary,
                                    []
                                ),
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
