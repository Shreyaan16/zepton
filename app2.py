import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import uuid
import io

# Set page configuration
st.set_page_config(
    page_title="NPI Finder PRO",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add a title and description
st.title("NPI Finder PRO")
st.markdown("""
This dashboard allows you to explore National Provider Identifier (NPI) data across different specialties and regions.
Upload your data file and select a specialty to visualize distribution patterns.
""")

# Data upload section
st.header("1. Upload Your Data")
uploaded_file = st.file_uploader(
    "Upload your Excel file containing NPI provider data (XLSX format)",
    type=["xlsx"]
)

# Show expected data schema
with st.expander("Show Expected Data Schema"):
    st.markdown("### Expected Data Format")
    st.markdown("""
    Your Excel file should contain the following columns:
    - **NPI** - National Provider Identifier
    - **Speciality** - Medical specialty of the provider
    - **State** - US state code (e.g., 'NY', 'CA') or full state name
    - **Region** - Geographic region (e.g., 'Northeast', 'Midwest', 'South', 'West')
    - **Usage Time (mins)** - Provider usage time in minutes
    
    Example data:
    """)
    
    # Create sample dataframe
    sample_data = {
        'NPI': ['1234567890', '2345678901', '3456789012', '4567890123', '5678901234'],
        'Speciality': ['Cardiology', 'Pediatrics', 'Oncology', 'Cardiology', 'Neurology'],
        'State': ['NY', 'CA', 'TX', 'FL', 'IL'],
        'Region': ['Northeast', 'West', 'South', 'South', 'Midwest'],
        'Usage Time (mins)': [45.2, 32.7, 58.1, 41.5, 37.8]
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df)

# Create a mapping of states to regions
region_mapping = {
    'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
    'Midwest': ['OH', 'MI', 'IN', 'IL', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
    'South': ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'],
    'West': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'WA', 'OR', 'CA', 'AK', 'HI']
}

# State codes to names mapping
state_code_to_name = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 
    'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 
    'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 
    'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 
    'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 
    'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 
    'NH': 'New Hampshire', 'NJ': 'New Jersey', '_NM': 'New Mexico', 
    'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 
    'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 
    'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 
    'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 
    'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 
    'WY': 'Wyoming', 'DC': 'District of Columbia'
}

# Reverse mapping
state_name_to_code = {v: k for k, v in state_code_to_name.items()}

# Process the uploaded data
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("Data loaded successfully!")
        
        # Display filters in the main area instead of sidebar
        st.header("2. Select Filters")
        
        # Create filter layout with three columns
        col1, col2, col3 = st.columns(3)
        
        # Get unique values for dropdowns
        specialties = sorted(df['Speciality'].unique())
        regions = sorted(df['Region'].unique())
        
        # Create dropdowns in the main area
        with col1:
            selected_specialty = st.selectbox("Select Specialty (Required)", specialties)
        
        with col2:
            selected_region = st.selectbox("Select Region", ["All Regions"] + regions)
        
        if selected_region == "All Regions":
            selected_region = None
        
        # Apply filters
        if selected_specialty:
            filtered_df = df[df['Speciality'] == selected_specialty]
            
            if selected_region:
                filtered_df = filtered_df[filtered_df['Region'] == selected_region]
                st.header(f"{selected_specialty} Providers in {selected_region} Region")
            else:
                st.header(f"{selected_specialty} Providers Across All Regions")
        else:
            st.warning("Please select a specialty to continue.")
            st.stop()
        
        # Display filtered dataset with download option
        st.subheader("Filtered Dataset")
        st.dataframe(filtered_df)
          # Add download buttons for the filtered data
        # For CSV download
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        
        # For Excel download
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            filtered_df.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_data = buffer.getvalue()
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Data as CSV",
                data=csv_data,
                file_name=f"{selected_specialty}_{'_' + selected_region if selected_region else ''}_data.csv",
                mime="text/csv"
            )
        
        with col2:
            st.download_button(
                label="Download Data as Excel",
                data=excel_data,
                file_name=f"{selected_specialty}_{'_' + selected_region if selected_region else ''}_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Display basic stats
        with st.expander("Dataset Overview", expanded=False):
            st.write("### Basic Statistics")
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            total_providers = len(filtered_df)
            total_states = filtered_df['State'].nunique()
            avg_usage = filtered_df['Usage Time (mins)'].mean()
            
            stat_col1.metric("Total Providers", total_providers)
            stat_col2.metric("States Represented", total_states)
            stat_col3.metric("Avg Usage Time (mins)", f"{avg_usage:.2f}")
            
            st.write("### Sample Data")
            st.dataframe(filtered_df.head(10))
        
        # Ensure state codes are consistent and only include states from df['State'].value_counts()
        def standardize_state_codes(df_to_process):
            data_df = df_to_process.copy()
            
            # Get valid states from the dataframe
            valid_states = df_to_process['State'].value_counts().index.tolist()
            
            # Handle both state names and state codes in the data
            if len(data_df) > 0:
                if len(data_df['State'].iloc[0]) > 2:  # Likely full state names
                    # Convert to state codes for consistency
                    data_df['StateCode'] = data_df['State'].map(lambda x: state_name_to_code.get(x, x))
                else:  # Likely state codes
                    data_df['StateCode'] = data_df['State']
                    # Also create a state name column for display
                    data_df['StateName'] = data_df['State'].map(lambda x: state_code_to_name.get(x, x))
                
                # Filter out invalid states
                data_df = data_df[data_df['StateCode'].isin(valid_states) | data_df['State'].isin(valid_states)]
                
                # Add region information
                def get_region(state_code):
                    for r, states in region_mapping.items():
                        if state_code in states:
                            return r
                    return "Unknown"
                
                data_df['MappedRegion'] = data_df['StateCode'].apply(get_region)
            
            return data_df
        
        # Process the dataframe
        processed_df = standardize_state_codes(filtered_df)
        
        # Create US Map visualization
        st.header("Geographic Distribution")
        tab1, tab2 = st.tabs(["Choropleth Map", "State Distribution"])
        
        with tab1:
            # Create state counts, only for states present in the data
            state_counts = processed_df.groupby('StateCode').size().reset_index(name='NPI_Count')
            state_counts['name'] = state_counts['StateCode'].map(lambda x: state_code_to_name.get(x, x))
            
            # Add region information for hover data
            def get_region(state_code):
                for r, states in region_mapping.items():
                    if state_code in states:
                        return r
                return "Unknown"
            
            state_counts['Region'] = state_counts['StateCode'].apply(get_region)
            
            # If a region is selected, filter to only show states in that region
            if selected_region:
                valid_states = region_mapping.get(selected_region, [])
                state_counts = state_counts[state_counts['StateCode'].isin(valid_states)]
                title_suffix = f" in {selected_region} Region"
            else:
                title_suffix = ""
            
            # Create a dataframe with only the states present in the data
            all_states_df = pd.DataFrame({
                'State': state_counts['StateCode'],
                'StateName': state_counts['name'],
                'NPI_Count': state_counts['NPI_Count'],
                'Region': state_counts['Region']
            })
            
            # Create choropleth map
            fig = px.choropleth(
                all_states_df,
                locations='State',
                locationmode='USA-states',
                color='NPI_Count',
                scope='usa',
                hover_name='StateName',
                hover_data=['Region', 'NPI_Count'],
                title=f'Distribution of {selected_specialty} NPIs Across US States{title_suffix}',
                color_continuous_scale='YlGnBu',
                labels={'NPI_Count': 'Number of NPIs', 'Region': 'Region'}
            )
            
            fig.update_layout(
                geo=dict(
                    showlakes=True,
                    lakecolor='rgb(255, 255, 255)',
                ),
                margin={"r":0,"t":50,"l":0,"b":0},
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add download button for the map
            st.download_button(
                label="Download Map as HTML",
                data=fig.to_html(),
                file_name=f"{selected_specialty}_{'_' + selected_region if selected_region else ''}_map.html",
                mime="text/html"
            )
        
        with tab2:
            # Create a bar chart of states
            state_bar = px.bar(
                state_counts.sort_values('NPI_Count', ascending=False),
                x='name',
                y='NPI_Count',
                title=f'Number of {selected_specialty} NPIs by State',
                color='NPI_Count',
                color_continuous_scale='YlGnBu',
                labels={'name': 'State', 'NPI_Count': 'Number of NPIs'}
            )
            
            state_bar.update_layout(
                xaxis={'categoryorder': 'total descending'},
                height=500
            )
            
            st.plotly_chart(state_bar, use_container_width=True)
        
        # Regional Analysis
        st.header("Regional Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Regional distribution
            if selected_region is None:
                region_counts = processed_df.groupby('MappedRegion').size().reset_index(name='NPI_Count')
                region_counts['Percentage'] = (region_counts['NPI_Count'] / region_counts['NPI_Count'].sum() * 100).round(1)
                
                region_pie = px.pie(
                    region_counts,
                    values='NPI_Count',
                    names='MappedRegion',
                    title=f'Regional Distribution of {selected_specialty} NPIs',
                    hole=0.3,
                    color_discrete_sequence=px.colors.sequential.YlGnBu
                )
                
                region_pie.update_traces(
                    textinfo='percent+label+value',
                    textposition='inside'
                )
                
                st.plotly_chart(region_pie, use_container_width=True)
            else:
                # If region is selected, show distribution of states within that region
                region_states = processed_df.groupby('StateCode').size().reset_index(name='NPI_Count')
                region_states['name'] = region_states['StateCode'].map(lambda x: state_code_to_name.get(x, x))
                
                state_pie = px.pie(
                    region_states,
                    values='NPI_Count',
                    names='name',
                    title=f'Distribution of {selected_specialty} NPIs in {selected_region} by State',
                    hole=0.3,
                    color_discrete_sequence=px.colors.sequential.YlGnBu
                )
                
                state_pie.update_traces(
                    textinfo='percent+label+value',
                    textposition='inside'
                )
                
                st.plotly_chart(state_pie, use_container_width=True)
        
        with col2:
            # Usage time analysis
            if 'Usage Time (mins)' in processed_df.columns:
                if selected_region is None:
                    # Regional usage analysis
                    usage_by_region = processed_df.groupby('MappedRegion')['Usage Time (mins)'].mean().reset_index()
                    usage_by_region['Usage Time (mins)'] = usage_by_region['Usage Time (mins)'].round(1)
                    
                    usage_fig = px.bar(
                        usage_by_region.sort_values('Usage Time (mins)', ascending=False),
                        x='MappedRegion',
                        y='Usage Time (mins)',
                        title=f'Average Usage Time for {selected_specialty} by Region',
                        color='Usage Time (mins)',
                        color_continuous_scale='YlGnBu',
                        text_auto='.1f'
                    )
                else:
                    # State usage analysis within region
                    usage_by_state = processed_df.groupby('StateCode')['Usage Time (mins)'].mean().reset_index()
                    usage_by_state['StateName'] = usage_by_state['StateCode'].map(lambda x: state_code_to_name.get(x, x))
                    usage_by_state['Usage Time (mins)'] = usage_by_state['Usage Time (mins)'].round(1)
                    
                    usage_fig = px.bar(
                        usage_by_state.sort_values('Usage Time (mins)', ascending=False),
                        x='StateName',
                        y='Usage Time (mins)',
                        title=f'Average Usage Time for {selected_specialty} by State in {selected_region}',
                        color='Usage Time (mins)',
                        color_continuous_scale='YlGnBu',
                        text_auto='.1f'
                    )
                
                usage_fig.update_layout(
                    height=400,
                    xaxis={'categoryorder': 'total descending'}
                )
                
                st.plotly_chart(usage_fig, use_container_width=True)
        
        # Advanced Analytics
        st.header("Advanced Analytics")
        
        # Usage time distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Usage Time Distribution
            fig = px.histogram(
                processed_df,
                x='Usage Time (mins)',
                nbins=20,
                title=f'Distribution of Usage Time for {selected_specialty}',
                color_discrete_sequence=['#3182bd']
            )
            
            fig.update_layout(
                xaxis_title='Usage Time (minutes)',
                yaxis_title='Number of Providers',
                showlegend=False
            )
            
            # Add vertical line for mean
            fig.add_vline(
                x=processed_df['Usage Time (mins)'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {processed_df['Usage Time (mins)'].mean():.1f} mins",
                annotation_position="top right"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box Plot by Region or State
            if selected_region is None:
                box_fig = px.box(
                    processed_df,
                    x='Region',
                    y='Usage Time (mins)',
                    title=f'Usage Time Distribution by Region for {selected_specialty}',
                    color='Region',
                    points='all'
                )
            else:
                box_fig = px.box(
                    processed_df,
                    x='State',
                    y='Usage Time (mins)',
                    title=f'Usage Time Distribution by State for {selected_specialty} in {selected_region}',
                    color='State',
                    points='all'
                )
            
            box_fig.update_layout(
                xaxis_title='Region' if selected_region is None else 'State',
                yaxis_title='Usage Time (minutes)',
                showlegend=False
            )
            
            st.plotly_chart(box_fig, use_container_width=True)
        
        # Additional insights
        st.header("Additional Insights")
        
        # Top States Table and Summary Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            # Top states with highest provider counts
            top_states = processed_df.groupby(['StateCode', 'State']).size().reset_index(name='Count')
            top_states['State Name'] = top_states['StateCode'].map(lambda x: state_code_to_name.get(x, x))
            top_states = top_states.sort_values('Count', ascending=False).head(10)
            
            st.subheader(f"Top 10 States for {selected_specialty} Providers")
            st.dataframe(
                top_states[['State Name', 'Count']].rename(columns={'Count': 'Number of Providers'}),
                use_container_width=True
            )
        
        with col2:
            # Summary Statistics for Usage Time
            st.subheader("Usage Time Statistics (minutes)")
            
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Minimum', 'Maximum', 'Standard Deviation'],
                'Value': [
                    f"{processed_df['Usage Time (mins)'].mean():.2f}",
                    f"{processed_df['Usage Time (mins)'].median():.2f}",
                    f"{processed_df['Usage Time (mins)'].min():.2f}",
                    f"{processed_df['Usage Time (mins)'].max():.2f}",
                    f"{processed_df['Usage Time (mins)'].std():.2f}"
                ]
            })
            
            st.dataframe(stats_df, use_container_width=True)
            
            # Add a gauge chart for average usage compared to overall average
            overall_avg = df['Usage Time (mins)'].mean()
            specialty_avg = processed_df['Usage Time (mins)'].mean()
            
            gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = specialty_avg,
                title = {'text': "Average Usage Time"},
                delta = {'reference': overall_avg, 'relative': False},
                gauge = {
                    'axis': {'range': [0, df['Usage Time (mins)'].max() * 1.2]},
                    'bar': {'color': "#2E86C1"},
                    'steps': [
                        {'range': [0, overall_avg * 0.7], 'color': "#D5F5E3"},
                        {'range': [overall_avg * 0.7, overall_avg * 1.3], 'color': "#ABEBC6"},
                        {'range': [overall_avg * 1.3, df['Usage Time (mins)'].max() * 1.2], 'color': "#82E0AA"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': overall_avg
                    }
                }
            ))
            
            gauge.update_layout(height=250)
            st.plotly_chart(gauge, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.stop()

else:
    # Display helpful message when no file is uploaded
    st.info("👆 Please upload an Excel file to begin analyzing your NPI provider data.")
    
    # Show example outputs with dummy data for preview
    st.header("Dashboard Preview")
    st.markdown("""
    This section shows example visualizations of what you'll see after uploading your data.
    The examples below use sample data and are for illustration purposes only.
    """)
    
    # Show sample choropleth map
    st.subheader("Example: Geographic Distribution")
 
    # Show a sample download button to demonstrate how users could export results
    st.download_button(
        label="Download Sample Report (CSV)",
        data="NPI,Specialty,State,Region,Usage Time (mins)\n1234567890,Cardiology,NY,Northeast,45.2\n2345678901,Pediatrics,CA,West,32.7",
        file_name="sample_npi_report.csv",
        mime="text/csv",
        disabled=True
    )

# Footer
st.markdown("---")
st.markdown("NPI Provider Visualization Dashboard | Created with Streamlit & Plotly")