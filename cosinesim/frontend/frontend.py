import streamlit as st
import requests
import json
from PIL import Image
import io
import base64

st.title("Idea Analysis Dashboard")

def display_results(response):
    # Display the plot
    image_bytes = base64.b64decode(response['plot'])
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, use_column_width=True)
    
    # Display results in tables
    for experiment in response['results']:
        st.header(experiment['title'])
        
        # Pool results
        st.subheader("Pool Results")
        pool_data = []
        for data in experiment['data']:
            pool_data.append({
                'Pool': data['file_path'],
                'Total Ideas': data['total'],
                'Unique Ideas': data['unique'],
                'Unique Fraction': f"{data['unique_fraction']:.3f}",
                'CI': f"({data['ci'][0]:.3f}, {data['ci'][1]:.3f})"
            })
        st.table(pool_data)
        
        # Comparisons
        st.subheader("Comparisons")
        comparison_data = []
        for comp in experiment['comparisons']:
            comparison_data.append({
                'Comparison': f"{comp['pool_1']} vs {comp['pool_2']}",
                'Difference': f"{comp['rd']:.3f}",
                'CI': f"({comp['ci'][0]:.3f}, {comp['ci'][1]:.3f})",
                'p-value': f"{comp['p_value']:.3f}",
                'z-stat': f"{comp['z_stat']:.3f}"
            })
        st.table(comparison_data)

# File upload section
st.header("Upload Experiment Files")
uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=['txt'])

if uploaded_files:
    # Create experiment configuration
    st.header("Configure Experiments")
    
    num_experiments = st.number_input("Number of experiments", min_value=1, max_value=10, value=1)
    
    experiments = []
    files_used = 0  # Track number of files assigned
    
    for i in range(num_experiments):
        st.subheader(f"Experiment {i+1}")
        
        title = st.text_input(f"Experiment {i+1} Title", f"Experiment {i+1}")
        
        # Let user select number of conditions for this experiment
        num_conditions = st.number_input(
            f"Number of conditions for experiment {i+1}", 
            min_value=2, 
            max_value=len(uploaded_files) - files_used,
            value=2
        )
        
        labels = []
        for j in range(num_conditions):
            label = st.text_input(
                f"Label for condition {j+1} in experiment {i+1}", 
                f"Condition {j+1}"
            )
            labels.append(label)
        
        if labels:
            experiments.append({
                "labels": labels,
                "title": title
            })
            files_used += len(labels)
    
    if st.button("Analyze"):
        try:
            # Prepare the multipart form data
            files = [
                ('files', (file.name, file, 'text/plain'))
                for file in uploaded_files
            ]
            
            # Prepare the configuration
            config_data = {
                "experiments": experiments
            }
            
            # Make request to backend
            response = requests.post(
                "http://localhost:8000/analyze",
                files=files,
                data={'config': json.dumps(config_data)}
            )
            
            if response.status_code == 200:
                display_results(response.json())
            else:
                st.error(f"Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with backend: {str(e)}")
        except Exception as e:
            st.error(f"Error processing results: {str(e)}")