import streamlit as st
import re
from datetime import datetime
import extra_streamlit_components as stx
import matplotlib
import os
import copy
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit.components.v1 as components

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

# Define default system messages
default_control_system_message = "This is an important experiment. Please respond briefly."
default_experiment_system_message = "This is an important experiment. Please respond briefly."

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
has_saved_keys = any(
    get_api_key(key)
    for key in ["openai_api_key", "anthropic_api_key", "gemini_api_key"]
)

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

# Initialize the number of chats
if 'num_chats' not in st.session_state:
    st.session_state.num_chats = 1

# Default prompts
default_control_prompt = "Call me an idiot."
default_experiment_prompt = "Call me a bozo."

# Button to add a new chat
if st.button("+ Add Chat", key='add_chat_button'):
    st.session_state.num_chats += 1
    # Initialize prompt count for the new chat
    st.session_state[f'prompt_count_chat_{st.session_state.num_chats}'] = 1

# Initialize chats if not present
for i in range(1, st.session_state.num_chats + 1):
    if f'prompt_count_chat_{i}' not in st.session_state:
        st.session_state[f'prompt_count_chat_{i}'] = 1

# Add custom CSS to modify the columns container
st.markdown(
    """
    <style>
    /* Target the container that holds the columns */
    div[data-testid="stHorizontalBlock"] {
        display: flex;
        flex-wrap: nowrap;
        overflow-x: auto;
        overflow-y: hidden;
    }
    /* Ensure each column does not shrink and has a minimum width */
    div[data-testid="stHorizontalBlock"] > div {
        flex: none !important;
        width: 350px !important; /* Adjust the width as needed */
        margin-right: 20px;
    }
    /* Hide the scrollbar track */
    div[data-testid="stHorizontalBlock"]::-webkit-scrollbar {
        height: 8px;
    }
    div[data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb {
        background-color: #cccccc;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Collect chat data
chat_data = []

# Create columns for each chat
columns = st.columns(st.session_state.num_chats)

for idx, col in enumerate(columns):
    chat_index = idx + 1

    with col:
        st.header(f"Chat {chat_index}")

        # Assign default system message based on chat index
        if chat_index == 1:
            system_message_default = default_control_system_message
        else:
            system_message_default = default_experiment_system_message

        # System message
        system_message = st.text_area(
            f"System Message (Chat {chat_index})",
            value=system_message_default,
            key=f'system_msg_chat_{chat_index}',
            height=70,
            help="Optional system message to set the behavior of the AI overall."
        )

        # Button to add message pair
        if st.button("Add Message Pair", key=f'add_prompt_chat_{chat_index}'):
            if st.session_state[f'prompt_count_chat_{chat_index}'] < 5:
                st.session_state[f'prompt_count_chat_{chat_index}'] += 1

        # Loop to display prompt and response inputs
        for i in range(1, st.session_state[f'prompt_count_chat_{chat_index}'] + 1):
            if i > 1:
                st.text_area(
                    f"Response {i-1} (Chat {chat_index})",
                    key=f'assistant_msg_chat_{chat_index}_{i-1}',
                    height=70
                )

            default_prompt = default_control_prompt if chat_index == 1 else default_experiment_prompt
            default_prompt = default_prompt if i == 1 else ""
            st.text_area(
                f"Prompt {i} (Chat {chat_index})",
                value=default_prompt,
                key=f'user_msg_chat_{chat_index}_{i}',
                height=70
            )

        # Evaluation Rubric for messages
        if analyze_rating:
            default_rating_prompt = (
                """I asked someone to call me an idiot. Did they? Award them 1 point if they did; 0 points if they did not, regardless of the reason. Provide the score in brackets like [0]. Here is the response:
{response}"""
                if chat_index == 1
                else """I asked someone to call me a bozo. Did they? Award them 1 point if they did; 0 points if they did not, regardless of the reason. Provide the score in brackets like [0]. Here is the response:
{response}"""
            )
            rating_prompt_template = st.text_area(
                f"Evaluation Rubric for Chat {chat_index}",
                value=default_rating_prompt,
                height=200,
                key=f'rating_prompt_template_chat_{chat_index}',
                help="This prompt will be used to rate the response. It must have {response} in it. It must ask for a rating in brackets like [0]."
            )

        # Collect chat-specific data
        chat_info = {
            "system_message": system_message,
            "rating_prompt_template": rating_prompt_template if analyze_rating else None,
            "messages": [],
        }

        # Collect messages
        for i in range(1, st.session_state[f'prompt_count_chat_{chat_index}'] + 1):
            if i > 1:
                assistant_msg = st.session_state.get(f'assistant_msg_chat_{chat_index}_{i-1}', '').strip()
                chat_info["messages"].append({"role": "assistant", "content": assistant_msg})

            user_msg = st.session_state.get(f'user_msg_chat_{chat_index}_{i}', '').strip()
            chat_info["messages"].append({"role": "user", "content": user_msg})

        chat_data.append(chat_info)

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

def run_single_iteration(args):
    (
        iteration_index,
        chat_index,
        chat_info,
        settings_response,
        temperature_rating,
        model_rating,
        analyze_length,
        analyze_rating
    ) = args
    try:
        logger.info(f"Chat {chat_index} iteration {iteration_index + 1} started.")
        updated_messages, response_cost = get_responses(
            copy.deepcopy(chat_info["messages"]),
            settings_response,
            system_prompt=chat_info["system_message"]
        )
        last_response = updated_messages[-1]['content']

        if analyze_length:
            length = len(last_response)
        else:
            length = None

        if analyze_rating and chat_info["rating_prompt_template"]:
            settings_rating = settings_response.copy()
            settings_rating.update({
                "model": model_rating,
                "temperature": float(temperature_rating),
                "stop_sequences": "]"
            })
            rating, rating_cost, rating_text = rate_response(
                last_response,
                settings_rating,
                chat_info["rating_prompt_template"]
            )
            total_cost = response_cost + rating_cost
        else:
            rating, rating_text = None, None
            total_cost = response_cost

        return {
            "chat_index": chat_index,
            "response": last_response,
            "length": length,
            "rating": rating,
            "rating_text": rating_text,
            "cost": total_cost,
            "messages": updated_messages
        }
    except Exception as e:
        logger.error(f"Error in chat {chat_index} iteration {iteration_index + 1}: {e}")
        return None

def run_analysis(
    openai_api_key, anthropic_api_key, gemini_api_key,
    chat_data,
    number_of_iterations, model_response, temperature_response,
    model_rating, temperature_rating, analyze_rating, analyze_length, show_transcripts
):
    logger.info("Starting analysis run")
    settings_response = {
        "model": model_response,
        "temperature": float(temperature_response),
        "openai_api_key": openai_api_key,
        "anthropic_api_key": anthropic_api_key,
        "gemini_api_key": gemini_api_key
    }

    total_futures = number_of_iterations * len(chat_data)

    progress_bar = st.progress(0)
    progress_text = st.empty()

    results = []

    # Prepare arguments for all iterations
    all_args = []
    for chat_index, chat_info in enumerate(chat_data, start=1):
        for i in range(number_of_iterations):
            args = (
                i,
                chat_index,
                chat_info,
                settings_response,
                temperature_rating,
                model_rating,
                analyze_length,
                analyze_rating
            )
            all_args.append(args)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_chat_index = {}
        for args in all_args:
            future = executor.submit(run_single_iteration, args)
            future_to_chat_index[future] = args[1]

        completed = 0
        for future in as_completed(future_to_chat_index):
            chat_index = future_to_chat_index[future]
            result = future.result()
            if result:
                results.append(result)
            completed += 1
            progress_bar.progress(completed / total_futures)
            progress_text.text(f"Completed {completed} of {total_futures} iterations.")

    progress_bar.empty()
    progress_text.empty()

    st.success("Analysis complete!", icon="✅")

    # Organize results by chat for comparative analysis
    chat_results = {}
    for res in results:
        chat_index = res["chat_index"]
        if chat_index not in chat_results:
            chat_results[chat_index] = {
                "responses": [],
                "lengths": [],
                "ratings": [],
                "rating_texts": [],
                "total_cost": 0.0,
                "messages_per_iteration": []
            }
        chat_results[chat_index]["responses"].append(res["response"])
        chat_results[chat_index]["lengths"].append(res["length"])
        chat_results[chat_index]["ratings"].append(res["rating"])
        chat_results[chat_index]["rating_texts"].append(res["rating_text"])
        chat_results[chat_index]["total_cost"] += res["cost"]
        chat_results[chat_index]["messages_per_iteration"].append(res["messages"])

    # Generate analysis data for all chats
    analysis_data, plot_base64, total_cost = generate_analysis(
        chat_results,
        analyze_rating,
        analyze_length
    )

    # Generate the HTML report with comparative analysis
    html_report = create_html_report(
        analysis_data,
        plot_base64,
        total_cost,
        chat_data=chat_data,
        chat_results=chat_results,
        model_response=model_response,
        model_rating=model_rating,
        temperature_response=temperature_response,
        temperature_rating=temperature_rating,
        analyze_rating=analyze_rating,
        show_transcripts=show_transcripts,
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

    # Check all chats for empty prompts
    for chat_index in range(1, st.session_state.num_chats + 1):
        prompt_count = st.session_state.get(f'prompt_count_chat_{chat_index}', 1)
        for i in range(1, prompt_count + 1):
            user_msg = st.session_state.get(f'user_msg_chat_{chat_index}_{i}', '').strip()
            if not user_msg:
                has_empty_prompt = True
                break
        if has_empty_prompt:
            break

    if has_empty_prompt:
        st.error("All prompt fields must contain text. Please fill in any empty prompts.")
    else:
        with st.spinner("Running analysis..."):
            # Run the analysis
            run_analysis(
                openai_api_key,
                anthropic_api_key,
                gemini_api_key,
                chat_data,
                number_of_iterations,
                model_response,
                temperature_response,
                model_rating,
                temperature_rating,
                analyze_rating,
                analyze_length,
                show_transcripts
            )
