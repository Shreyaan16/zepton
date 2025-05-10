import streamlit as st
import requests
import time
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
from streamlit_lottie import st_lottie

st.set_page_config(layout="wide", page_title="NPI Survey Analysis", page_icon="📊")

# App state management
if 'initial_load' not in st.session_state:
    st.session_state.initial_load = True
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False
if 'npi_df' not in st.session_state:
    st.session_state.npi_df = None
if 'survey_df' not in st.session_state:
    st.session_state.survey_df = None
if 'rf_model' not in st.session_state:
    st.session_state.rf_model = None
if 'animation_complete' not in st.session_state:
    st.session_state.animation_complete = False

# Function to load Lottie animation
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load lottie animations
lottie_medical = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
lottie_analysis = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_XyoSty.json")
lottie_processing = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kseho6rf.json")

# Function to convert hour:minute to minutes since midnight
def to_minutes(hour, minute):
    return hour * 60 + minute

# Function to calculate active time window considering midnight spanning
def calculate_active_window(row):
    login_mins = to_minutes(row['login_hour'], row['login_minute'])
    logout_mins = to_minutes(row['logout_hour'], row['logout_minute'])
    login_date = datetime.strptime(row['login_date'], "%Y-%m-%d")
    logout_date = datetime.strptime(row['logout_date'], "%Y-%m-%d")
    
    if logout_date > login_date or (logout_date == login_date and logout_mins < login_mins):
        active_time = (1440 - login_mins) + logout_mins
    else:
        active_time = logout_mins - login_mins
    
    return login_mins, logout_mins, active_time

# Preprocess npi.csv: Extract time patterns and verify usage time
def preprocess_npi_data(npi_df):
    npi_df[['login_mins', 'logout_mins', 'calculated_active_time']] = npi_df.apply(
        lambda row: pd.Series(calculate_active_window(row)), axis=1
    )
    
    npi_df['usage_time_valid'] = npi_df.apply(
        lambda row: abs(row['calculated_active_time'] - row['Usage Time (mins)']) <= 5, axis=1
    )
    
    return npi_df

# Function to check if an NPI is active in a given time slot
def is_active_in_timeslot(row, target_time, window_size=60):
    target_mins = to_minutes(target_time[0], target_time[1])
    
    half_window = window_size // 2
    slot_start = max(0, target_mins - half_window)
    slot_end = min(1439, target_mins + half_window)
    
    login_mins = row['login_mins']
    logout_mins = row['logout_mins']
    
    if logout_mins < login_mins:
        if slot_start <= 1439 and login_mins <= slot_end:
            return True
        if slot_end >= 0 and logout_mins >= slot_start:
            return True
        return False
    else:
        return max(login_mins, slot_start) <= min(logout_mins, slot_end)

# Extract features for the model
def extract_features(row, target_time):
    target_mins = to_minutes(target_time[0], target_time[1])
    
    hour = target_time[0]
    minute = target_time[1]
    
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    is_morning = 1 if 6 <= hour < 12 else 0
    is_afternoon = 1 if 12 <= hour < 18 else 0
    is_evening = 1 if 18 <= hour < 24 else 0
    is_night = 1 if 0 <= hour < 6 else 0
    
    survey_attempts_normalized = row['Count of Survey Attempts'] / 10.0
    usage_time_normalized = row['Usage Time (mins)'] / 240.0
    
    region_features = [
        row.get('Region_Midwest', 0),
        row.get('Region_Northeast', 0),
        row.get('Region_South', 0),
        row.get('Region_West', 0)
    ]
    
    specialty_features = [
        row.get('Speciality_Cardiology', 0),
        row.get('Speciality_General Practice', 0),
        row.get('Speciality_Neurology', 0),
        row.get('Speciality_Oncology', 0),
        row.get('Speciality_Orthopedics', 0),
        row.get('Speciality_Pediatrics', 0),
        row.get('Speciality_Radiology', 0)
    ]
    
    login_mins = row['login_mins']
    logout_mins = row['logout_mins']
    
    if logout_mins < login_mins:
        if login_mins <= target_mins:
            time_since_login = target_mins - login_mins
            time_until_logout = (1440 - target_mins) + logout_mins
        else:
            time_since_login = (1440 - login_mins) + target_mins
            time_until_logout = logout_mins - target_mins
    else:
        if login_mins <= target_mins <= logout_mins:
            time_since_login = target_mins - login_mins
            time_until_logout = logout_mins - target_mins  
        elif target_mins < login_mins:
            time_since_login = -1 * (login_mins - target_mins)
            time_until_logout = (logout_mins - login_mins) + abs(time_since_login)
        else:
            time_until_logout = -1 * (target_mins - logout_mins)
            time_since_login = (logout_mins - login_mins) + abs(time_until_logout)
    
    time_since_login_normalized = time_since_login / 1440.0
    time_until_logout_normalized = time_until_logout / 1440.0
    
    inside_window = 1 if is_active_in_timeslot(row, target_time) else 0
    
    return [hour_sin, hour_cos, is_morning, is_afternoon, is_evening, is_night,
            survey_attempts_normalized, usage_time_normalized, 
            time_since_login_normalized, time_until_logout_normalized, inside_window] + region_features + specialty_features

# Generate training data for the Random Forest model
def generate_training_data(npi_df, survey_df):
    X_train = []
    y_train = []
    
    npi_survey_map = {}
    for _, survey_row in survey_df.iterrows():
        npi = survey_row['NPI']
        attempt_time = (survey_row['attempt_hour'], survey_row['attempt_minute'])
        
        if npi not in npi_survey_map:
            npi_survey_map[npi] = []
        
        npi_survey_map[npi].append(attempt_time)
    
    for _, row in npi_df.iterrows():
        npi = row['NPI']
        survey_times = npi_survey_map.get(npi, [])
        
        for hour in range(0, 24, 6):  # Reduced resolution for performance
            for minute in [0]:
                target_time = (hour, minute)
                features = extract_features(row, target_time)
                
                participated = 0
                for s_time in survey_times:
                    s_mins = to_minutes(s_time[0], s_time[1])
                    t_mins = to_minutes(target_time[0], target_time[1])
                    if abs(s_mins - t_mins) <= 30:
                        participated = 1
                        break
                
                X_train.append(features)
                y_train.append(participated)
    
    return np.array(X_train), np.array(y_train)

# Train the Random Forest model on NPI survey participation patterns
def train_rf_model(npi_df, survey_df):
    X, y = generate_training_data(npi_df, survey_df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using smaller model for performance
    rf_model = RandomForestClassifier(
        n_estimators=5,
        max_depth=8,
        min_samples_split=5,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Random Forest Model Accuracy: {accuracy:.4f}")
    
    return rf_model

# Main function to analyze survey participation
def analyze_survey_participation(survey_id, time_str, survey_df, npi_df, rf_model):
    survey_row = survey_df[survey_df['Survey ID'] == survey_id]
    if survey_row.empty:
        return "Survey ID not found."
    
    try:
        hh, mm = map(int, time_str.split(':'))
        if hh < 0 or hh > 23 or mm < 0 or mm > 59:
            return "Invalid time format. Please use HH:MM in 24-hour format."
        target_time = (hh, mm)
    except:
        return "Invalid time format. Please use HH:MM in 24-hour format."
    
    survey_participants = set(survey_df[survey_df['Survey ID'] == survey_id]['NPI'].tolist())
    
    active_npis = []
    
    for _, row in npi_df.iterrows():
        npi = row['NPI']
        
        is_active = is_active_in_timeslot(row, target_time)
        
        if is_active:
            participated = npi in survey_participants
            
            if participated:
                participation_prob = 1.0
            else:
                features = extract_features(row, target_time)
                participation_prob = rf_model.predict_proba([features])[0][1]
            
            region = "Unknown"
            if row.get('Region_Midwest', 0) == 1:
                region = "Midwest"
            elif row.get('Region_Northeast', 0) == 1:
                region = "Northeast"
            elif row.get('Region_South', 0) == 1:
                region = "South"
            elif row.get('Region_West', 0) == 1:
                region = "West"
            
            state_columns = [col for col in row.index if col.startswith('State_')]
            state = "Unknown"
            for state_col in state_columns:
                if row[state_col] == 1:
                    state = state_col.replace('State_', '')
                    break
            
            specialty_columns = [col for col in row.index if col.startswith('Speciality_')]
            specialty = "Unknown"
            for specialty_col in specialty_columns:
                if row[specialty_col] == 1:
                    specialty = specialty_col.replace('Speciality_', '')
                    break
            
            active_npis.append({
                'NPI': npi,
                'Participated': participated,
                'Participation Probability': participation_prob,
                'Survey Attempts History': row['Count of Survey Attempts'],
                'Usage Time': row['Usage Time (mins)'],
                'Active Window': f"{row['login_hour']:02d}:{row['login_minute']:02d} to {row['logout_hour']:02d}:{row['logout_minute']:02d}",
                'Region': region,
                'State': state,
                'Specialty': specialty
            })
    
    active_npis_sorted = sorted(active_npis, key=lambda x: x['Participation Probability'], reverse=True)
    
    participants_count = sum(1 for npi in active_npis if npi['Participated'])
    active_npi_count = len(active_npis)
    
    participation_percentage = (participants_count / active_npi_count * 100) if active_npi_count > 0 else 0
    
    output = {
        'Survey ID': survey_id,
        'Analysis Time': time_str,
        'Total NPIs in Database': len(npi_df),
        'Active NPIs at Analysis Time': active_npi_count,
        'Survey Participants Among Active NPIs': participants_count,
        'Participation Percentage': participation_percentage,
        'Active NPIs with Participation Probability': active_npis_sorted
    }
    
    return output

# Function to analyze active NPIs at different times
def analyze_active_npis_by_time(npi_df):
    time_counts = {}
    for hour in range(0, 24, 2):  # Reduced resolution for performance 
        for minute in [0]:
            target_time = (hour, minute)
            time_str = f"{hour:02d}:{minute:02d}"
            
            active_count = sum(1 for _, row in npi_df.iterrows() if is_active_in_timeslot(row, target_time))
            time_counts[time_str] = active_count
    
    return time_counts

# Create visualizations for region, state, and specialty distributions
def create_visualizations(active_npi_data):
    df = pd.DataFrame(active_npi_data)
    
    region_counts = df['Region'].value_counts().reset_index()
    region_counts.columns = ['Region', 'Count']
    
    fig_region = px.bar(
        region_counts, 
        x='Region', 
        y='Count',
        title='Active NPIs by Region',
        color='Region',
        labels={'Count': 'Number of Active NPIs'},
        height=400
    )
    
    state_counts = df['State'].value_counts().reset_index()
    state_counts.columns = ['State', 'Count']
    state_counts = state_counts.head(10)  # Reduced for performance
    
    fig_state = px.bar(
        state_counts, 
        x='State', 
        y='Count',
        title='Active NPIs by State (Top 10)',
        color='State',
        labels={'Count': 'Number of Active NPIs'},
        height=400
    )
    
    specialty_counts = df['Specialty'].value_counts().reset_index()
    specialty_counts.columns = ['Specialty', 'Count']
    
    fig_specialty = px.pie(
        specialty_counts, 
        values='Count', 
        names='Specialty',
        title='Active NPIs by Specialty',
        height=400
    )
    
    region_participation = df.groupby('Region')['Participated'].agg(['sum', 'count']).reset_index()
    region_participation['Participation Rate'] = (region_participation['sum'] / region_participation['count'] * 100).round(2)
    region_participation.columns = ['Region', 'Participants', 'Total', 'Participation Rate (%)']
    
    fig_region_participation = px.bar(
        region_participation,
        x='Region',
        y='Participation Rate (%)',
        title='Participation Rate by Region',
        color='Region',
        height=400
    )
    
    specialty_participation = df.groupby('Specialty')['Participated'].agg(['sum', 'count']).reset_index()
    specialty_participation['Participation Rate'] = (specialty_participation['sum'] / specialty_participation['count'] * 100).round(2)
    specialty_participation.columns = ['Specialty', 'Participants', 'Total', 'Participation Rate (%)']
    
    fig_specialty_participation = px.bar(
        specialty_participation,
        x='Specialty',
        y='Participation Rate (%)',
        title='Participation Rate by Specialty',
        color='Specialty',
        height=400
    )
    
    return fig_region, fig_state, fig_specialty, fig_region_participation, fig_specialty_participation

def main():
    # Simplified CSS for a lightweight UI
    st.markdown("""
    <style>
        .main-header {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            color: #6B46C1;
            background: linear-gradient(to right, #6B46C1, #D53F8C);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient 3s ease infinite;
            background-size: 200% auto;
        }
        @keyframes gradient {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        .sub-header {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #D53F8C;
        }
        .card {
            background: #f8f9fa;
            border-left: 4px solid #6B46C1;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
            color: #333;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 10px rgba(0,0,0,0.2);
        }
        .graph-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .graph-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(107, 70, 193, 0.2);
        }
        .success-animation {
            margin: 0 auto;
            display: block;
        }
        .fade-in {
            animation: fadeIn 1s ease-in-out;
        }
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        .stButton > button {
            background: linear-gradient(90deg, #6B46C1, #D53F8C) !important;
            color: white !important;
            border: none !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            transform: scale(1.05) !important;
            box-shadow: 0 0 15px rgba(107, 70, 193, 0.4) !important;
        }
        .progress-bar {
            width: 100%;
            height: 5px;
            background: linear-gradient(90deg, #6B46C1, #D53F8C);
            animation: progress 2s ease-in-out infinite;
        }
        /* Progress Bar Animation */
        @keyframes progress {
            0% { width: 0; }
            100% { width: 100%; }
        }
        .progress-bar {
            height: 4px;
            background-color: #6B46C1;
            margin-bottom: 20px;
            animation: progress 2s ease-out;
        }
    </style>
    """, unsafe_allow_html=True)    # Header
    st.markdown("<div class='main-header'>HCP Campaign Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; margin-bottom: 30px;'>Analyze NPI participation patterns with interactive visualization</div>", unsafe_allow_html=True)

    # Initial loading animation
    if st.session_state.initial_load and not st.session_state.animation_complete:
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st_lottie(lottie_medical, height=200, key="loading")
            st.markdown("<div style='text-align: center; margin-top: 20px; font-size: 20px;'>Initializing dashboard...</div>", unsafe_allow_html=True)
            time.sleep(2)
        loading_placeholder.empty()
        st.session_state.animation_complete = True
        st.session_state.initial_load = False
        st.rerun()

    # File upload section
    if not st.session_state.show_analysis:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='sub-header'>NPI Data</div>", unsafe_allow_html=True)
            npi_file = st.file_uploader("Upload NPI CSV file", type=['csv'], key="npi_uploader")
            if npi_file is not None:
                try:
                    npi_df = pd.read_csv(npi_file)
                    if npi_df.empty:
                        st.error("The uploaded NPI CSV file is empty.")
                    else:
                        st.session_state.npi_df = preprocess_npi_data(npi_df)
                        st.success("NPI data loaded successfully!")
                        # Add progress bar animation
                        st.markdown("<div class='progress-bar'></div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error processing NPI CSV: {str(e)}")
        
        with col2:
            st.markdown("<div class='sub-header'>Survey Data</div>", unsafe_allow_html=True)
            survey_file = st.file_uploader("Upload Survey CSV file", type=['csv'], key="survey_uploader")
            if survey_file is not None:
                try:
                    survey_df = pd.read_csv(survey_file)
                    if survey_df.empty:
                        st.error("The uploaded Survey CSV file is empty.")
                    else:
                        st.session_state.survey_df = survey_df
                        st.success("Survey data loaded successfully!")
                        # Add progress bar animation
                        st.markdown("<div class='progress-bar'></div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error processing Survey CSV: {str(e)}")
          # Process data button
        if st.session_state.npi_df is not None and st.session_state.survey_df is not None:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Process Data and Show Analysis", key="process_data"):
                    process_placeholder = st.empty()
                    with process_placeholder.container():
                        st_lottie(lottie_processing, height=150, key="processing")
                        st.markdown("<div style='text-align: center; color: #6B46C1; font-weight: bold;'>Processing data...</div>", unsafe_allow_html=True)
                        # Train the Random Forest model
                        st.session_state.rf_model = train_rf_model(st.session_state.npi_df, st.session_state.survey_df)
                        time.sleep(1)
                        st.success("Analysis Ready!")
                        time.sleep(1)
                    process_placeholder.empty()
                    st.session_state.show_analysis = True
                    st.rerun()
      # Analysis section
    if st.session_state.show_analysis:
        # Ensure we have the trained model
        if st.session_state.rf_model is None:
            with st.spinner("Training model..."):
                st.session_state.rf_model = train_rf_model(st.session_state.npi_df, st.session_state.survey_df)
        
        # Add a small animation when entering analysis view
        st.markdown("""
        <div class="fade-in" style="text-align:center; margin-bottom:20px;">
            <div style="font-size:18px; color:#6B46C1; font-weight:bold; margin-bottom:10px;">Analysis Dashboard Ready</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 1, 3])
        with col2:
            st_lottie(lottie_analysis, height=100, key="analysis_icon", speed=0.7)
        
        # Reset button
        col1, col2, col3 = st.columns([3, 1, 3])
        with col2:
            if st.button("Upload New Data Files", key="reset_analysis"):
                reset_placeholder = st.empty()
                with reset_placeholder.container():
                    st.success("Resetting app state...")
                    time.sleep(1)
                st.session_state.show_analysis = False
                st.session_state.npi_df = None
                st.session_state.survey_df = None
                st.session_state.rf_model = None
                st.rerun()
        
        # Analysis parameters
        st.markdown("<div class='sub-header'>Analysis Parameters</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            survey_id = st.number_input("Survey ID", min_value=100000, max_value=999999, value=100010, key="survey_id")
        
        with col2:
            time_str = st.text_input("Analysis Time (HH:MM)", value="08:30", key="time_input")
            run_button = st.button("Run Analysis", key="run_button")
        
        # Tabs for different analysis views
        tab1, tab2, tab3 = st.tabs(["📈 Survey Analysis", "🌍 NPI Distribution", "⏰ Time Patterns"])
        
        with tab1:
            if run_button or ('run_analysis_triggered' in st.session_state and st.session_state.run_analysis_triggered):
                with st.spinner("Analyzing survey participation..."):
                    result = analyze_survey_participation(survey_id, time_str, st.session_state.survey_df, st.session_state.npi_df, st.session_state.rf_model)
                
                st.session_state.run_analysis_triggered = False
                
                if isinstance(result, str):
                    st.error(result)
                else:
                    # Success animation
                    success_placeholder = st.empty()
                    with success_placeholder.container():
                        st.markdown("""
                        <div style='text-align:center; margin:10px 0;'>
                            <div style='font-size:20px; color:#6B46C1; font-weight:bold;'>Analysis Complete!</div>
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(0.5)
                    
                    st.markdown("<div class='sub-header'>Analysis Summary</div>", unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                            <div class='card'>
                                <h4>Total NPIs</h4>
                                <p style='font-size: 20px;'>{result['Total NPIs in Database']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                            <div class='card'>
                                <h4>Active NPIs</h4>
                                <p style='font-size: 20px;'>{result['Active NPIs at Analysis Time']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                            <div class='card'>
                                <h4>Participation</h4>
                                <p style='font-size: 20px;'>{result['Participation Percentage']:.2f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    if result['Active NPIs with Participation Probability']:
                        fig_region, fig_state, fig_specialty, fig_region_part, fig_specialty_part = create_visualizations(
                            result['Active NPIs with Participation Probability']
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
                            st.plotly_chart(fig_region, use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
                            st.plotly_chart(fig_state, use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
                            st.plotly_chart(fig_specialty, use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
                            st.plotly_chart(fig_region_part, use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
                        st.plotly_chart(fig_specialty_part, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    results_df = pd.DataFrame(result['Active NPIs with Participation Probability'])
                    st.markdown("<div class='sub-header'>Active NPIs</div>", unsafe_allow_html=True)
                    st.dataframe(results_df, use_container_width=True)
                    
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name=f"survey_{survey_id}_analysis_{time_str.replace(':', '')}.csv",
                        mime='text/csv',
                    )
            
            if run_button:
                st.session_state.run_analysis_triggered = True
                st.rerun()
        
        with tab2:
            st.markdown("<div class='sub-header'>Overall NPI Distribution</div>", unsafe_allow_html=True)
            
            with st.spinner("Generating NPI distribution..."):
                region_data = {}
                state_data = {}
                specialty_data = {}
                
                for _, row in st.session_state.npi_df.iterrows():
                    region = "Unknown"
                    for region_name in ['Midwest', 'Northeast', 'South', 'West']:
                        if row.get(f'Region_{region_name}', 0) == 1:
                            region = region_name
                            break
                    region_data[region] = region_data.get(region, 0) + 1
                    
                    state = "Unknown"
                    state_columns = [col for col in row.index if col.startswith('State_')]
                    for state_col in state_columns:
                        if row[state_col] == 1:
                            state = state_col.replace('State_', '')
                            break
                    state_data[state] = state_data.get(state, 0) + 1
                    
                    specialty = "Unknown"
                    specialty_columns = [col for col in row.index if col.startswith('Speciality_')]
                    for specialty_col in specialty_columns:
                        if row[specialty_col] == 1:
                            specialty = specialty_col.replace('Speciality_', '')
                            break
                    specialty_data[specialty] = specialty_data.get(specialty, 0) + 1
                
                region_df = pd.DataFrame([{'Region': k, 'Count': v} for k, v in region_data.items()])
                state_df = pd.DataFrame([{'State': k, 'Count': v} for k, v in state_data.items()]).sort_values('Count', ascending=False).head(10)
                specialty_df = pd.DataFrame([{'Specialty': k, 'Count': v} for k, v in specialty_data.items()])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
                    fig_region_all = px.bar(region_df, x='Region', y='Count', color='Region', height=400)
                    st.plotly_chart(fig_region_all, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
                    fig_state_all = px.bar(state_df, x='State', y='Count', color='State', height=400)
                    st.plotly_chart(fig_state_all, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
                    fig_specialty_all = px.pie(specialty_df, values='Count', names='Specialty', height=400)
                    st.plotly_chart(fig_specialty_all, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
        
        with tab3:
            st.markdown("<div class='sub-header'>NPI Activity by Time</div>", unsafe_allow_html=True)
            
            with st.spinner("Analyzing time patterns..."):
                time_counts = analyze_active_npis_by_time(st.session_state.npi_df)
                time_df = pd.DataFrame([{'Time': k, 'Active NPIs': v} for k, v in time_counts.items()])
                
                login_hour_counts = st.session_state.npi_df['login_hour'].value_counts().reset_index()
                login_hour_counts.columns = ['Hour', 'Count']
                login_hour_counts = login_hour_counts.sort_values('Hour')
                
                logout_hour_counts = st.session_state.npi_df['logout_hour'].value_counts().reset_index()
                logout_hour_counts.columns = ['Hour', 'Count']
                logout_hour_counts = logout_hour_counts.sort_values('Hour')
                
                st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
                fig_time = px.line(time_df, x='Time', y='Active NPIs', markers=True, height=400)
                st.plotly_chart(fig_time, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
                    fig_login = px.bar(login_hour_counts, x='Hour', y='Count', height=400)
                    st.plotly_chart(fig_login, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
                    fig_logout = px.bar(logout_hour_counts, x='Hour', y='Count', height=400)
                    st.plotly_chart(fig_logout, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
