import streamlit as st
import re
from datetime import datetime
import extra_streamlit_components as stx
import matplotlib
import os
import copy
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

from call_gpt import call_gpt
from log_love import setup_logging
logger = None
logger = setup_logging(logger)
from analysis import generate_analysis, create_html_report

# For the plots to display correctly in Streamlit
matplotlib.use('Agg')

# Define maximum number of workers
MAX_WORKERS = 10  # Adjust as needed for parallel iterations

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

def get_responses(messages, settings_response, system_prompt=None):
    total_steps = len(messages)
    logger.info(f"Fetching responses for {total_steps} messages:")
    logger.info(messages)
    completed_messages = []
    total_response_cost = 0.0

    for message in messages:
        if message['role'] == 'user':
            completed_messages.append(message)
        elif message['role'] == 'assistant':
            if message['content'].strip():
                completed_messages.append(message)
            else:
                response, response_cost = call_gpt(
                    completed_messages.copy(),
                    settings=settings_response,
                    return_pricing=True,
                    system_prompt=system_prompt
                )
                completed_messages.append({"role": "assistant", "content": response})
                total_response_cost += response_cost  # Accumulate response cost

    # Final assistant response
    response, response_cost = call_gpt(
        completed_messages.copy(),
        settings=settings_response,
        return_pricing=True,
        system_prompt=system_prompt
    )
    completed_messages.append({"role": "assistant", "content": response})
    total_response_cost += response_cost  # Accumulate final response cost

    return completed_messages, total_response_cost  # Return both messages and cost

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
    help="Select the AI model for generating responses."
)

# Temperature for response generation
temperature_response = st.sidebar.slider(
    "Temperature for Response Generation", min_value=0.0, max_value=1.0, value=1.0, step=0.1,
    help="Controls the randomness of the response generation."
)

# Model for rating
model_rating = st.sidebar.selectbox(
    "Model for Rating", AVAILABLE_MODELS, index=0,
    help="Select the AI model for rating the responses."
)

# Checkboxes for analysis options
analyze_rating = st.sidebar.checkbox("Use AI to analyze ratings", value=True)
analyze_length = st.sidebar.checkbox("Analyze length of response", value=False)
show_transcripts = st.sidebar.checkbox("Add a table of all responses", value=True)

# Temperature for rating
temperature_rating = st.sidebar.slider(
    "Temperature for Rating", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
    help="Controls the randomness of the rating generation."
)

# Initialize message counts
max_message_pairs = 5

# Main area for prompts
col1, col2 = st.columns(2)
default_control_prompt = "Call me an idiot."
default_experiment_prompt = "Call me a bozo."

with col1:
    st.header("Control Message")
    if 'prompt_count_ctrl' not in st.session_state:
        st.session_state.prompt_count_ctrl = 1

    # Add system message field for control
    control_system_message = st.text_area(
        "System Message (Control)",
        value="",
        key='system_msg_ctrl',
        height=70,
        help="Optional system message to set the behavior of the AI overall. Example: 'Be terse. This is serious."
    )

    if st.button("Add Message Pair", key='add_prompt_ctrl'):
        if st.session_state.prompt_count_ctrl < max_message_pairs:
            st.session_state.prompt_count_ctrl += 1

    # Loop to display prompt and response inputs
    for i in range(1, st.session_state.prompt_count_ctrl + 1):
        if i > 1:
            st.text_area(f"Response {i-1} (Control)", key=f'assistant_msg_ctrl_{i-1}', height=70)

        st.text_area(f"Prompt {i} (Control)",
                     value=default_control_prompt if i == 1 else "",
                     key=f'user_msg_ctrl_{i}',
                     height=70)

    # Evaluation Rubric for control messages. MAKE SURE TO INCLUDE {{response}} IN THE PROMPT. The double curly braces are important!!!
    if analyze_rating:
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
    if 'prompt_count_exp' not in st.session_state:
        st.session_state.prompt_count_exp = 1

    # Add system message field for experiment
    experiment_system_message = st.text_area(
        "System Message (Experiment)",
        value="",
        key='system_msg_exp',
        height=70,
        help="Optional system message to set the behavior for experiment messages."
    )

    if st.button("Add Message Pair", key='add_prompt_exp'):
        if st.session_state.prompt_count_exp < max_message_pairs:
            st.session_state.prompt_count_exp += 1

    # Loop to display prompt and response inputs
    for i in range(1, st.session_state.prompt_count_exp + 1):
        if i > 1:
            st.text_area(f"Response {i-1} (Experiment)", key=f'assistant_msg_exp_{i-1}', height=70)

        st.text_area(f"Prompt {i} (Experiment)",
                     value=default_experiment_prompt if i == 1 else "",
                     key=f'user_msg_exp_{i}',
                     height=70)

    # Evaluation Rubric for experiment messages
    if analyze_rating:
        experiment_rating_prompt_template = st.text_area(
            "Evaluation Rubric for Experiment Messages",
            value="""I asked someone to call me a bozo. Did they? Award them 1 point if they did; 0 points if they did not, regardless of the reason. Provide the score in brackets like [0]. Here is the response:
{response}""",
            height=200,
            key='experiment_rating_prompt_template',
            help="Template used by the model to rate the responses for the experiment prompt."
        )

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

def run_single_iteration_control(args):
    (
        iteration_index,
        messages_ctrl_original,
        settings_response,
        control_rating_prompt_template,
        temperature_rating,
        model_rating,
        control_system_message
    ) = args
    try:
        logger.info(f"Control iteration {iteration_index + 1} started.")
        updated_messages_ctrl, response_cost_ctrl = get_responses(copy.deepcopy(messages_ctrl_original), settings_response, system_prompt=control_system_message)
        last_response_ctrl = updated_messages_ctrl[-1]['content']

        if analyze_length:
            length = len(last_response_ctrl)
        else:
            length = None

        if analyze_rating:
            settings_rating = settings_response.copy()
            settings_rating.update({
                "model": model_rating,
                "temperature": float(temperature_rating),
                "stop_sequences": "]"
            })
            rating_ctrl, rating_cost_ctrl, rating_text_ctrl = rate_response(last_response_ctrl, settings_rating, control_rating_prompt_template)
        else:
            rating_ctrl, rating_cost_ctrl, rating_text_ctrl = None, 0.0, None

        return {
            "response": last_response_ctrl,
            "length": length,
            "rating": rating_ctrl,
            "rating_text": rating_text_ctrl,
            "cost": response_cost_ctrl + (rating_cost_ctrl if analyze_rating else 0.0),
            "messages": updated_messages_ctrl
        }
    except Exception as e:
        logger.error(f"Error in control iteration {iteration_index + 1}: {e}")
        return None

def run_single_iteration_experiment(args):
    (
        iteration_index,
        messages_exp_original,
        settings_response,
        experiment_rating_prompt_template,
        temperature_rating,
        model_rating,
        experiment_system_message
    ) = args
    try:
        logger.info(f"Experiment iteration {iteration_index + 1} started.")
        updated_messages_exp, response_cost_exp = get_responses(copy.deepcopy(messages_exp_original), settings_response, system_prompt=experiment_system_message)
        last_response_exp = updated_messages_exp[-1]['content']

        if analyze_length:
            length = len(last_response_exp)
        else:
            length = None

        if analyze_rating:
            settings_rating = settings_response.copy()
            settings_rating.update({
                "model": model_rating,
                "temperature": float(temperature_rating),
                "stop_sequences": "]"
            })
            rating_exp, rating_cost_exp, rating_text_exp = rate_response(last_response_exp, settings_rating, experiment_rating_prompt_template)
        else:
            rating_exp, rating_cost_exp, rating_text_exp = None, 0.0, None

        return {
            "response": last_response_exp,
            "length": length,
            "rating": rating_exp,
            "rating_text": rating_text_exp,
            "cost": response_cost_exp + (rating_cost_exp if analyze_rating else 0.0),
            "messages": updated_messages_exp
        }
    except Exception as e:
        logger.error(f"Error in experiment iteration {iteration_index + 1}: {e}")
        return None

def run_analysis(
    openai_api_key, anthropic_api_key, gemini_api_key,
    messages_ctrl_original, messages_exp_original,
    control_rating_prompt_template, experiment_rating_prompt_template,
    number_of_iterations, model_response, temperature_response,
    model_rating, temperature_rating, analyze_rating, analyze_length, show_transcripts,
    control_system_message=None, experiment_system_message=None
):
    logger.info("Starting analysis run")
    if not messages_ctrl_original:
        st.error("Please provide at least one message for the control prompt.")
        return
    if not messages_exp_original:
        st.error("Please provide at least one message for the experiment prompt.")
        return

    # Settings for response generation
    settings_response = {
        "model": model_response,
        "temperature": float(temperature_response),
        "openai_api_key": openai_api_key,
        "anthropic_api_key": anthropic_api_key,
        "gemini_api_key": gemini_api_key
    }

    # Initialize lists to store per-iteration messages
    messages_ctrl_per_iteration = []
    messages_exp_per_iteration = []

    # Prepare arguments for control iterations
    control_args = [
        (
            i,
            messages_ctrl_original,
            settings_response,
            control_rating_prompt_template,
            temperature_rating,
            model_rating,
            control_system_message
        )
        for i in range(number_of_iterations)
    ]

    # Prepare arguments for experiment iterations
    experiment_args = [
        (
            i,
            messages_exp_original,
            settings_response,
            experiment_rating_prompt_template,
            temperature_rating,
            model_rating,
            experiment_system_message
        )
        for i in range(number_of_iterations)
    ]

    total_futures = len(control_args) + len(experiment_args)

    progress_bar = st.progress(0)
    progress_text = st.empty()

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_type = {}
        for args in control_args:
            future = executor.submit(run_single_iteration_control, args)
            future_to_type[future] = 'control'
        for args in experiment_args:
            future = executor.submit(run_single_iteration_experiment, args)
            future_to_type[future] = 'experiment'

        completed = 0
        for future in as_completed(future_to_type):
            iteration_type = future_to_type[future]
            result = future.result()
            if result:
                if iteration_type == 'control':
                    messages_ctrl_per_iteration.append(result["messages"])
                    results.append(('control', result))
                else:
                    messages_exp_per_iteration.append(result["messages"])
                    results.append(('experiment', result))
            completed += 1
            progress_bar.progress(completed / total_futures)
            progress_text.text(f"Completed {completed} of {total_futures} iterations.")

    # Initialize total costs
    total_cost_ctrl = sum(r[1]["cost"] for r in results if r[0] == 'control')
    total_cost_exp = sum(r[1]["cost"] for r in results if r[0] == 'experiment')

    # Separate responses and other metrics
    responses_ctrl = [r[1]["response"] for r in results if r[0] == 'control']
    lengths_ctrl = [r[1]["length"] for r in results if r[0] == 'control']
    ratings_ctrl = [r[1]["rating"] for r in results if r[0] == 'control']
    rating_texts_ctrl = [r[1]["rating_text"] for r in results if r[0] == 'control']

    responses_exp = [r[1]["response"] for r in results if r[0] == 'experiment']
    lengths_exp = [r[1]["length"] for r in results if r[0] == 'experiment']
    ratings_exp = [r[1]["rating"] for r in results if r[0] == 'experiment']
    rating_texts_exp = [r[1]["rating_text"] for r in results if r[0] == 'experiment']

    total_cost = total_cost_ctrl + total_cost_exp

    progress_bar.empty()
    progress_text.empty()

    st.success("Analysis complete!", icon="✅")

    # Generate analysis data
    analysis_data, plot_base64, _ = generate_analysis(
        responses_ctrl,
        lengths_ctrl,
        ratings_ctrl,
        total_cost_ctrl,
        responses_exp,
        lengths_exp,
        ratings_exp,
        total_cost_exp,
        analyze_rating,
        analyze_length,
    )

    # Generate the HTML report
    logger.info("Creating HTML report")
    html_report = create_html_report(
        analysis_data,
        plot_base64,
        total_cost,
        messages_ctrl_original,
        messages_exp_original,
        messages_ctrl_per_iteration,
        messages_exp_per_iteration,
        control_rating_prompt_template,
        experiment_rating_prompt_template,
        analyze_rating=analyze_rating,
        show_transcripts=show_transcripts,
        responses_ctrl=responses_ctrl,
        responses_exp=responses_exp,
        ratings_ctrl=ratings_ctrl,
        ratings_exp=ratings_exp,
        rating_texts_ctrl=rating_texts_ctrl,
        rating_texts_exp=rating_texts_exp,
        model_response=model_response,
        model_rating=model_rating,
        temperature_response=temperature_response,
        temperature_rating=temperature_rating,
        control_system_message=control_system_message,
        experiment_system_message=experiment_system_message,
    )

    st.download_button(
        label="Download Report as HTML",
        data=html_report,
        file_name="analysis_report.html",
        mime="text/html"
    )

    # Display the HTML report in Streamlit
    st.components.v1.html(html_report, height=1000, scrolling=True)

# Run Analysis Streamlit UI
if st.button("Run Analysis", key="run_analysis_button", type="primary"):
    has_empty_prompt = False

    # Check control prompts for empty fields
    for i in range(1, st.session_state.prompt_count_ctrl + 1):
        if not st.session_state.get(f'user_msg_ctrl_{i}', '').strip():
            has_empty_prompt = True
            break

    # Check experiment prompts for empty fields
    if not has_empty_prompt:
        for i in range(1, st.session_state.prompt_count_exp + 1):
            if not st.session_state.get(f'user_msg_exp_{i}', '').strip():
                has_empty_prompt = True
                break

    # Collect the original messages for control
    messages_ctrl_original = []
    for i in range(1, st.session_state.prompt_count_ctrl + 1):
        # First add the assistant message from the previous round if it exists
        if i > 1:
            assistant_msg = st.session_state.get(f'assistant_msg_ctrl_{i-1}', '').strip()
            assistant_message = {"role": "assistant", "content": assistant_msg}
            messages_ctrl_original.append(assistant_message)
        
        # Then add the user message for this round
        user_msg = st.session_state.get(f'user_msg_ctrl_{i}', '').strip()
        user_message = {"role": "user", "content": user_msg}
        messages_ctrl_original.append(user_message)

    # Collect the original messages for experiment
    messages_exp_original = []
    if not has_empty_prompt:
        for i in range(1, st.session_state.prompt_count_exp + 1):
            # First add the assistant message from the previous round if it exists
            if i > 1:
                assistant_msg = st.session_state.get(f'assistant_msg_exp_{i-1}', '').strip()
                assistant_message = {"role": "assistant", "content": assistant_msg}
                messages_exp_original.append(assistant_message)
            
            # Then add the user message for this round
            user_msg = st.session_state.get(f'user_msg_exp_{i}', '').strip()
            user_message = {"role": "user", "content": user_msg}
            messages_exp_original.append(user_message)

    if has_empty_prompt:
        st.error("All prompt fields must contain text. Please fill in any empty prompts.")
    else:
        with st.spinner("Running analysis..."):
            # Run the analysis
            run_analysis(
                openai_api_key,
                anthropic_api_key,
                gemini_api_key,
                messages_ctrl_original,
                messages_exp_original,
                control_rating_prompt_template if analyze_rating else None,
                experiment_rating_prompt_template if analyze_rating else None,
                number_of_iterations,
                model_response,
                temperature_response,
                model_rating,
                temperature_rating,
                analyze_rating,
                analyze_length,
                show_transcripts,
                control_system_message=control_system_message,
                experiment_system_message=experiment_system_message
            )
