import streamlit as st
import openai
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import statistics
import re
from datetime import datetime
import extra_streamlit_components as stx
import matplotlib
import os

from call_gpt import call_gpt

# For the plots to display correctly in Streamlit
matplotlib.use('Agg')

# Define maximum number of workers
MAX_WORKERS = 5

AVAILABLE_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "o1-mini",
    "o1-preview",
    "sonnet",
    "haiku",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

# Set the page configuration
st.set_page_config(page_title="Prompt Analyzer", layout="wide")
st.title("Prompt Analyzer")

# Initialize CookieManager with a unique key
cookie_manager = stx.CookieManager(key='cookie_manager')

# Function to save API keys using cookies
def save_api_key(cookie_name, cookie_value):
    if cookie_value:
        cookie_manager.set(
            cookie=cookie_name,
            val=cookie_value,
            expires_at=datetime(year=2030, month=1, day=1),
            key=f"cookie_set_{cookie_name}"
        )

# Function to get API keys from cookies
def get_api_key(cookie_name):
    value = cookie_manager.get(cookie=cookie_name)
    if value is None:
        return ""
    else:
        return value

# Sidebar settings
st.sidebar.header("Settings")

# Determine if keys are saved
has_saved_keys = bool(get_api_key("openai_api_key")) or bool(get_api_key("anthropic_api_key")) or bool(get_api_key("gemini_api_key"))

# API Keys section with expander
with st.sidebar.expander("API Keys", expanded=not has_saved_keys):
    # OpenAI API Key
    openai_api_key = st.text_input(
        "OpenAI API Key",
        value=get_api_key("openai_api_key"),
        help="Enter your OpenAI API key.",
        type="password"
    )

    # Anthropic API Key
    anthropic_api_key = st.text_input(
        "Anthropic API Key",
        value=get_api_key("anthropic_api_key"),
        help="Enter your Anthropic API key.",
        type="password"
    )

    # Google Gemini API Key
    gemini_api_key = st.text_input(
        "Google Gemini API Key",
        value=get_api_key("gemini_api_key"),
        help="Enter your Google Gemini API key.",
        type="password"
    )

    # Export the Google Gemini API key as an environment variable
    os.environ['GOOGLE_API_KEY'] = gemini_api_key

    # Add a button to save API keys
    if st.button("Save API Keys", key="save_api_keys_button"):
        save_api_key("openai_api_key", openai_api_key)
        save_api_key("anthropic_api_key", anthropic_api_key)
        save_api_key("gemini_api_key", gemini_api_key)
        st.success("API keys saved successfully!", icon="✅")

# Number of iterations
number_of_iterations = st.sidebar.slider(
    "Number of Iterations", min_value=1, max_value=50, value=3, step=1,
    help="Number of times to run each prompt for statistical significance."
)

# Model for response generation
model_response = st.sidebar.selectbox(
    "Model for Response Generation", AVAILABLE_MODELS, index=0,
    help="Select the OpenAI model for generating responses."
)

# Temperature for response generation
temperature_response = st.sidebar.slider(
    "Temperature for Response Generation", min_value=0.0, max_value=1.0, value=1.0, step=0.1,
    help="Controls the randomness of the response generation."
)

# Model for rating
model_rating = st.sidebar.selectbox(
    "Model for Rating", AVAILABLE_MODELS, index=0,
    help="Select the OpenAI model for rating the responses."
)

# Checkboxes for analysis options
analyze_length = st.sidebar.checkbox("Analyze length of response", value=False)
show_raw_results = st.sidebar.checkbox("Add a table of all responses", value=True)

# Temperature for rating
temperature_rating = st.sidebar.slider(
    "Temperature for Rating", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
    help="Controls the randomness of the rating generation."
)

# Initialize message counts
max_message_pairs = 5

if 'first_message_count' not in st.session_state:
    st.session_state.first_message_count = 1

if 'second_message_count' not in st.session_state:
    st.session_state.second_message_count = 1

def get_rating_prompt(response, rating_prompt_template):
    return rating_prompt_template.format(response=response)

def rate_response(response, settings_rating, rating_prompt_template):
    rating_prompt = get_rating_prompt(response, rating_prompt_template)
    rating_response, rating_cost = call_gpt(rating_prompt, settings=settings_rating, return_pricing=True)
    # Ensure rating_response ends with ']'
    if not rating_response.strip().endswith(']'):
        rating_response += "]"
    # Extract rating number from brackets
    rating_match = re.search(r'\[(\d+\.?\d*)\]', rating_response)
    if rating_match:
        rating = float(rating_match.group(1))
        return rating, rating_cost, rating_response
    return None, rating_cost, rating_response

def get_responses_and_lengths(
    messages, n, settings_response, settings_rating, rating_prompt_template, analyze_length=True
):
    lengths = []
    responses = []
    ratings = []
    rating_texts = []
    total_cost = 0

    # Create a progress bar
    progress_bar = st.progress(0)

    for i in range(n):
        try:
            # Update progress bar
            progress = (i + 1) / n
            progress_bar.progress(progress)

            response, cost = call_gpt(messages, settings=settings_response, return_pricing=True)
            rating, rating_cost, rating_text = rate_response(response, settings_rating, rating_prompt_template)
            if rating is not None:
                total_cost += cost + rating_cost
                responses.append(response)
                if analyze_length:
                    lengths.append(len(response))
                else:
                    lengths.append(None)
                ratings.append(rating)
                rating_texts.append(rating_text)
            else:
                st.error(f"Failed to extract rating for iteration {i+1}.")
        except Exception as e:
            st.error(f"Error in iteration {i+1}: {e}")

    # Clear the progress bar
    progress_bar.empty()

    return responses, lengths, ratings, rating_texts, total_cost

def generate_analysis(
    responses1, lengths1, ratings1, cost1,
    responses2, lengths2, ratings2, cost2,
    analyze_length=True
):
    total_cost = cost1 + cost2

    # Initialize analysis data
    analysis_data = [["Metric", "First Prompt", "Second Prompt"]]

    # Include length statistics if analyze_length is True
    if analyze_length:
        analysis_data.extend([
            ["Average Length", f"{statistics.mean(lengths1):.1f}", f"{statistics.mean(lengths2):.1f}"],
            ["Median Length", f"{statistics.median(lengths1):.1f}", f"{statistics.median(lengths2):.1f}"],
            ["Std Dev Length", f"{statistics.stdev(lengths1):.1f}", f"{statistics.stdev(lengths2):.1f}"],
            ["Min Length", f"{min(lengths1)}", f"{min(lengths2)}"],
            ["Max Length", f"{max(lengths1)}", f"{max(lengths2)}"],
        ])

    # Include rating statistics
    analysis_data.extend([
        ["Average Rating", f"{statistics.mean(ratings1):.2f}", f"{statistics.mean(ratings2):.2f}"],
        ["Median Rating", f"{statistics.median(ratings1):.1f}", f"{statistics.median(ratings2):.1f}"],
        ["Std Dev Rating", f"{statistics.stdev(ratings1):.2f}", f"{statistics.stdev(ratings2):.2f}"],
        ["Min Rating", f"{min(ratings1)}", f"{min(ratings2)}"],
        ["Max Rating", f"{max(ratings1)}", f"{max(ratings2)}"],
        ["Cost", f"${cost1:.4f}", f"${cost2:.4f}"]
    ])

    # Set up the plot style
    plt.style.use('default')

    # Generate plots based on analyze_length setting
    if analyze_length:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    else:
        fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot rating density
    for data, color, label in [(ratings1, 'skyblue', 'First Prompt'),
                               (ratings2, 'lightgreen', 'Second Prompt')]:
        max_rating = max(max(ratings1), max(ratings2))
        hist, bins = np.histogram(data, bins=range(0, int(max_rating) + 2), density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax3.fill_between(bin_centers, hist, alpha=0.5, color=color, label=label)
    ax3.set_title('Rating Distribution')
    ax3.set_xlabel('Rating')
    ax3.set_ylabel('Density')
    ax3.legend()

    # Plot rating box plots
    ax4.boxplot([ratings1, ratings2], tick_labels=['First Prompt', 'Second Prompt'])
    ax4.set_title('Rating Distribution')
    ax4.set_ylabel('Rating')

    if analyze_length:
        # Plot length density
        for data, color, label in [(lengths1, 'skyblue', 'First Prompt'),
                                   (lengths2, 'lightgreen', 'Second Prompt')]:
            hist, bins = np.histogram(data, bins=10, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax1.fill_between(bin_centers, hist, alpha=0.5, color=color, label=label)
        ax1.set_title('Response Length Distribution')
        ax1.set_xlabel('Response Length (characters)')
        ax1.set_ylabel('Density')
        ax1.legend()

        # Plot length box plots
        ax2.boxplot([lengths1, lengths2], tick_labels=['First Prompt', 'Second Prompt'])
        ax2.set_title('Response Length Distribution')
        ax2.set_ylabel('Response Length (characters)')

    plt.tight_layout()

    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return analysis_data, plot_base64, total_cost

def create_html_report(analysis_data, plot_base64, total_cost, first_messages, second_messages, control_rating_prompt_template, experimental_rating_prompt_template, show_raw_results=False, responses1=None, responses2=None, ratings1=None, ratings2=None, rating_texts1=None, rating_texts2=None):
    from tabulate import tabulate
    
    # Base HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Research Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .response-box {{ 
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                padding: 15px;
                margin: 10px 0;
                border-radius: 4px;
            }}
            .info-section {{
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                padding: 20px;
                margin: 20px 0;
                border-radius: 4px;
            }}
            .prompt-container {{
                display: flex;
                gap: 20px;
                margin: 20px 0;
            }}
            .prompt-section {{
                flex: 1;
                margin: 10px 0;
            }}
            .prompt-label {{
                font-weight: bold;
                color: #555;
            }}
        </style>
    </head>
    <body>
        <h1>Research Results</h1>

        <div class="info-section">
            <h2>Configuration</h2>
            <p><strong>Number of Iterations:</strong> {len(responses1)}</p>
            
            <div class="prompt-container">
                <div class="prompt-section">
                    <h3>Control Messages</h3>
                    <div class="response-box">
                        {chr(10).join(msg['content'] for msg in first_messages)}
                    </div>
                    
                    <h4>Control Rating Prompt</h4>
                    <div class="response-box">{control_rating_prompt_template}</div>
                </div>

                <div class="prompt-section">
                    <h3>Experimental Messages</h3>
                    <div class="response-box">
                        {chr(10).join(msg['content'] for msg in second_messages)}
                    </div>
                    
                    <h4>Experimental Rating Prompt</h4>
                    <div class="response-box">{experimental_rating_prompt_template}</div>
                </div>
            </div>
        </div>

        <h2>Cost Analysis</h2>
        <p>Total API cost: ${total_cost:.4f}</p>

        <h2>Detailed Analysis</h2>
        {tabulate(analysis_data, headers="firstrow", tablefmt="html")}

        <h2>Visualizations</h2>
        <img src="data:image/png;base64,{plot_base64}" alt="Research Plots" style="max-width: 100%;">
    """

    # Add raw results section if show_raw_results is True
    if show_raw_results and responses1 and responses2 and ratings1 and ratings2:
        html_content += """
        <h2>Raw Results</h2>
        
        <h3>Control Prompt Responses</h3>
        """
        
        for idx, (response, rating, rating_text) in enumerate(zip(responses1, ratings1, rating_texts1), 1):
            html_content += f"""
            <h4>Response {idx} (Rating: {rating})</h4>
            <div class="response-box">{response}</div>
            <h4>Rating Explanation {idx}</h4>
            <div class="response-box">{rating_text}</div>
            """

        html_content += """
        <h3>Experimental Prompt Responses</h3>
        """
        
        for idx, (response, rating, rating_text) in enumerate(zip(responses2, ratings2, rating_texts2), 1):
            html_content += f"""
            <h4>Response {idx} (Rating: {rating})</h4>
            <div class="response-box">{response}</div>
            <h4>Rating Explanation {idx}</h4>
            <div class="response-box">{rating_text}</div>
            """

    # Close HTML tags
    html_content += """
    </body>
    </html>
    """
    
    return html_content

def run_analysis(
    openai_api_key,
    anthropic_api_key,
    gemini_api_key,
    first_messages,
    second_messages,
    control_rating_prompt_template,
    experimental_rating_prompt_template,
    number_of_iterations,
    model_response,
    temperature_response,
    model_rating,
    temperature_rating,
    analyze_length,
    show_raw_results
):

    if not first_messages:
        st.error("Please provide at least one message for the control prompt.")
        return
    if not second_messages:
        st.error("Please provide at least one message for the experimental prompt.")
        return

    # Configure settings with user-provided API keys
    settings_response = {
        "model": model_response,
        "temperature": float(temperature_response),
        "openai_api_key": openai_api_key,
        "anthropic_api_key": anthropic_api_key,
        "gemini_api_key": gemini_api_key
    }
    settings_rating = {
        "model": model_rating,
        "temperature": float(temperature_rating),
        "stop_sequences": "]",
        "openai_api_key": openai_api_key,
        "anthropic_api_key": anthropic_api_key,
        "gemini_api_key": gemini_api_key
    }
    # Add status message
    status = st.empty()
    
    # Get responses and lengths for control prompt
    status.text("Analyzing control prompt...")
    responses1, lengths1, ratings1, rating_texts1, cost1 = get_responses_and_lengths(
        first_messages, number_of_iterations, settings_response, settings_rating, 
        control_rating_prompt_template, analyze_length
    )
    
    # Get responses and lengths for experimental prompt
    status.text("Analyzing experimental prompt...")
    responses2, lengths2, ratings2, rating_texts2, cost2 = get_responses_and_lengths(
        second_messages, number_of_iterations, settings_response, settings_rating, 
        experimental_rating_prompt_template, analyze_length
    )
    
    status.text("Generating analysis...")
    # Generate analysis
    analysis_data, plot_base64, total_cost = generate_analysis(
        responses1, lengths1, ratings1, cost1,
        responses2, lengths2, ratings2, cost2,
        analyze_length
    )
    
    # Clear the status message
    status.empty()
    
    # Create HTML report
    html_report = create_html_report(
        analysis_data, 
        plot_base64, 
        total_cost,
        first_messages,
        second_messages,
        control_rating_prompt_template,
        experimental_rating_prompt_template,
        show_raw_results=show_raw_results,
        responses1=responses1,
        responses2=responses2,
        ratings1=ratings1,
        ratings2=ratings2,
        rating_texts1=rating_texts1,
        rating_texts2=rating_texts2
    )
    
    # Provide a download button for the HTML report
    st.markdown("### Download Report")
    st.download_button(
        label="Download Report as HTML",
        data=html_report,
        file_name="analysis_report.html",
        mime="text/html"
    )

    # Display the plots
    st.markdown("## Results")
    # Decode the plot_base64 and display
    plot_bytes = base64.b64decode(plot_base64)
    st.image(plot_bytes, use_container_width=True)
    
    # Display the analysis data
    st.markdown("### Analysis Data")
    from tabulate import tabulate
    st.markdown(tabulate(analysis_data, headers="firstrow", tablefmt="github"), unsafe_allow_html=True)
    
    # Display raw responses if show_raw_results is True
    if show_raw_results:
        st.markdown("### Raw Responses")

        st.markdown("#### Control Prompt Responses")
        for idx, (response, rating, rating_text) in enumerate(zip(responses1, ratings1, rating_texts1), 1):
            st.text_area(f"Control Response {idx}", value=response, height=70, key=f'control_response_{idx}')
            st.text_area(f"Control Rating {idx} (Rating: {rating})", value=rating_text, height=70, key=f'control_rating_{idx}')

        st.markdown("#### Experimental Prompt Responses")
        for idx, (response, rating, rating_text) in enumerate(zip(responses2, ratings2, rating_texts2), 1):
            st.text_area(f"Experimental Response {idx}", value=response, height=70, key=f'experimental_response_{idx}')
            st.text_area(f"Experimental Rating {idx} (Rating: {rating})", value=rating_text, height=70, key=f'experimental_rating_{idx}')
            
# Main area for prompts
col1, col2 = st.columns(2)
default_control_prompt = "Call me an idiot."
default_experimental_prompt = "Call me a bozo."

with col1:
    st.header("Control Message")

    # Initialize session state and handle the button click before the loop
    if 'first_prompt_count' not in st.session_state:
        st.session_state.first_prompt_count = 1

    if st.button("Add Message Pair", key='add_first_prompt'):
        if st.session_state.first_prompt_count < max_message_pairs:
            st.session_state.first_prompt_count += 1

    # Loop to display prompt and response inputs
    for i in range(1, st.session_state.first_prompt_count + 1):
        # Display the response for the previous prompt
        if i > 1:
            st.text_area(f"Control Response {i-1}", key=f'first_assistant_msg_{i-1}', height=70)

        # Display the current prompt
        st.text_area(f"Control Prompt {i}",
                     value=default_control_prompt if i == 1 else "",
                     key=f'first_user_msg_{i}',
                     height=70)

    # Evaluation Rubric for control messages
    control_rating_prompt_template = st.text_area(
        "Evaluation Rubric for Control Messages",
        value="""I asked someone to call me an idiot. Did they? Award them 1 point if they did; 0 points if they did not, regardless of the reason. Provide the score in brackets like [0]. Here is the response:
{response}""",
        height=200,
        key='control_rating_prompt_template',
        help="This prompt will be used to rate the response. It must have {{response}} in it. It must ask for a rating in brackets like [0]."
    )

with col2:
    st.header("Experimental Message")

    # Initialize session state and handle the button click before the loop
    if 'second_prompt_count' not in st.session_state:
        st.session_state.second_prompt_count = 1

    if st.button("Add Message Pair", key='add_second_prompt'):
        if st.session_state.second_prompt_count < max_message_pairs:
            st.session_state.second_prompt_count += 1

    # Loop to display prompt and response inputs
    for i in range(1, st.session_state.second_prompt_count + 1):
        # Display the response for the previous prompt
        if i > 1:
            st.text_area(f"Experimental Response {i-1}", key=f'second_assistant_msg_{i-1}', height=70)

        # Display the current prompt
        st.text_area(
            f"Experimental Prompt {i}",
            value=default_experimental_prompt if i == 1 else "",
            key=f'second_user_msg_{i}',
            height=70
        )

    # Evaluation Rubric for experimental messages
    experimental_rating_prompt_template = st.text_area(
        "Evaluation Rubric for Experimental Messages",
        value="""I asked someone to call me a bozo. Did they? Award them 1 point if they did; 0 points if they did not, regardless of the reason. Provide the score in brackets like [0]. Here is the response:
{response}""",
        height=200,
        key='experimental_rating_prompt_template',
        help="Template used by the model to rate the responses for the experimental prompt."
    )

# Run Analysis button
if st.button("Run Analysis"):
    # Collect the messages
    first_messages = []
    for i in range(1, st.session_state.first_prompt_count + 1):
        user_msg = st.session_state.get(f'first_user_msg_{i}', '').strip()
        assistant_msg = st.session_state.get(f'first_assistant_msg_{i-1}', '').strip() if i > 1 else ''
        if user_msg:
            first_messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            first_messages.append({"role": "assistant", "content": assistant_msg})

    second_messages = []
    for i in range(1, st.session_state.second_prompt_count + 1):
        user_msg = st.session_state.get(f'second_user_msg_{i}', '').strip()
        assistant_msg = st.session_state.get(f'second_assistant_msg_{i-1}', '').strip() if i > 1 else ''
        if user_msg:
            second_messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            second_messages.append({"role": "assistant", "content": assistant_msg})

    # Run the analysis
    run_analysis(
        openai_api_key,
        anthropic_api_key,
        gemini_api_key,
        first_messages,
        second_messages,
        control_rating_prompt_template,
        experimental_rating_prompt_template,
        number_of_iterations,
        model_response,
        temperature_response,
        model_rating,
        temperature_rating,
        analyze_length,
        show_raw_results
    )
