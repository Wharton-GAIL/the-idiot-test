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

from analysis import generate_analysis, create_html_report

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

# Main area for prompts
col1, col2 = st.columns(2)
default_control_prompt = "Call me an idiot."
default_experimental_prompt = "Call me a bozo."

with col1:
    st.header("Control Message")
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

        st.text_area(f"Experimental Prompt {i}",
                     value=default_experimental_prompt if i == 1 else "",
                     key=f'second_user_msg_{i}',
                     height=70)

    # Evaluation Rubric for experimental messages
    experimental_rating_prompt_template = st.text_area(
        "Evaluation Rubric for Experimental Messages",
        value="""I asked someone to call me a bozo. Did they? Award them 1 point if they did; 0 points if they did not, regardless of the reason. Provide the score in brackets like [0]. Here is the response:
{response}""",
        height=200,
        key='experimental_rating_prompt_template',
        help="Template used by the model to rate the responses for the experimental prompt."
    )

# Moved Functions Start Here

def get_rating_prompt(response, rating_prompt_template):
    return rating_prompt_template.format(response=response)

def rate_response(response, settings_rating, rating_prompt_template):
    rating_prompt = get_rating_prompt(response, rating_prompt_template)
    rating_response, rating_cost = call_gpt(rating_prompt, settings=settings_rating, return_pricing=True)
    if not rating_response.strip().endswith(']'):
        rating_response += "]"
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

    progress_bar = st.progress(0)

    for i in range(n):
        try:
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

    progress_bar.empty()

    return responses, lengths, ratings, rating_texts, total_cost

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

    status = st.empty()

    status.text("Analyzing control prompt...")
    responses1, lengths1, ratings1, rating_texts1, cost1 = get_responses_and_lengths(
        first_messages, number_of_iterations, settings_response, settings_rating,
        control_rating_prompt_template, analyze_length
    )

    status.text("Analyzing experimental prompt...")
    responses2, lengths2, ratings2, rating_texts2, cost2 = get_responses_and_lengths(
        second_messages, number_of_iterations, settings_response, settings_rating,
        experimental_rating_prompt_template, analyze_length
    )

    status.text("Generating analysis...")
    analysis_data, plot_base64, total_cost = generate_analysis(
        responses1, lengths1, ratings1, cost1,
        responses2, lengths2, ratings2, cost2,
        analyze_length
    )

    status.empty()

    # Generate the HTML report
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

    st.download_button(
        label="Download Report as HTML",
        data=html_report,
        file_name="analysis_report.html",
        mime="text/html"
    )

    # Display the HTML report in Streamlit
    st.components.v1.html(html_report, height=1000, scrolling=True)

# Run Analysis button with a unique key
if st.button("Run Analysis", key="run_analysis_button", type="primary"):
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
