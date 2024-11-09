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
st.set_page_config(page_title="GPT Prompt Comparison Tool", layout="wide")
st.title("GPT Prompt Comparison Tool")

# Initialize CookieManager with a unique key
cookie_manager = stx.CookieManager(key='cookie_manager')

# Function to save API keys using cookies
def save_api_key(cookie_name, cookie_value):
    if cookie_value:
        cookie_manager.set(
            cookie=cookie_name,
            val=cookie_value,
            expires_at=datetime(year=2030, month=1, day=1)
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
has_saved_keys = bool(get_api_key("openai_api_key")) or bool(get_api_key("anthropic_api_key"))

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

    # Add a button to save API keys
    if st.button("Save API Keys"):
        save_api_key("openai_api_key", openai_api_key)
        save_api_key("anthropic_api_key", anthropic_api_key)
        st.success("API keys saved successfully!")

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

# Temperature for rating
temperature_rating = st.sidebar.slider(
    "Temperature for Rating", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
    help="Controls the randomness of the rating generation."
)

# Rating prompt template
rating_prompt_template = st.sidebar.text_area(
    "Rating Prompt Template", value="""I asked someone to call me a bozo. Did they? Award them 1 point if they did; 0 points if they did not, regardless of the reason:
<RESPONSE>
{response}
</RESPONSE>
Provide the score in brackets like [0].
""", height=200,
    help="Template used by the model to rate the responses. Instruct it to provide a score in brackets like [1]."
)

# Initialize message counts
max_message_pairs = 5

if 'first_message_count' not in st.session_state:
    st.session_state.first_message_count = 1

if 'second_message_count' not in st.session_state:
    st.session_state.second_message_count = 1

# Function to call OpenAI API
def call_gpt(prompt_or_messages, settings, return_pricing=False):
    openai.api_key = settings.get("api_key")
    model = settings.get("model", "gpt-3.5-turbo")
    temperature = settings.get("temperature", 1.0)
    stop_sequences = settings.get("stop_sequences", None)

    # Check if prompt_or_messages is a list of messages or a string prompt
    if isinstance(prompt_or_messages, str):
        # Use 'prompt' parameter (Not recommended for ChatGPT models)
        response = openai.Completion.create(
            engine=model,
            prompt=prompt_or_messages,
            temperature=temperature,
            max_tokens=1000,  # Adjust as needed
            stop=stop_sequences,
        )
        text = response.choices[0].text.strip()
    else:
        # Use 'messages' parameter
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt_or_messages,
            temperature=temperature,
            stop=stop_sequences,
        )
        text = response.choices[0].message['content'].strip()

    # Optionally return pricing information
    cost = 0  # Placeholder for cost calculation
    if return_pricing:
        # Pricing info per https://openai.com/pricing
        # Note: Prices are per token
        if 'usage' in response:
            prompt_tokens = response['usage']['prompt_tokens']
            completion_tokens = response['usage']['completion_tokens']
            # Now calculate cost based on model
            model_pricing = {
                'gpt-3.5-turbo': {'prompt': 0.0015, 'completion': 0.002},
                'gpt-3.5-turbo-16k': {'prompt': 0.003, 'completion': 0.004},
                'gpt-4': {'prompt': 0.03, 'completion': 0.06},
                'gpt-4-32k': {'prompt': 0.06, 'completion': 0.12},
            }
            pricing = model_pricing.get(model, {'prompt': 0.0015, 'completion': 0.002})
            cost = (prompt_tokens * pricing['prompt'] / 1000) + (completion_tokens * pricing['completion'] / 1000)
        else:
            cost = 0
        return text, cost
    else:
        return text

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
        return rating, rating_cost
    return None, rating_cost

def get_responses_and_lengths(messages, n, settings_response, settings_rating, rating_prompt_template):
    lengths = []
    responses = []
    ratings = []
    total_cost = 0
    for i in range(n):
        try:
            response, cost = call_gpt(messages, settings=settings_response, return_pricing=True)
            rating, rating_cost = rate_response(response, settings_rating, rating_prompt_template)
            if rating is not None:
                total_cost += cost + rating_cost
                responses.append(response)
                lengths.append(len(response))
                ratings.append(rating)
            else:
                st.error(f"Failed to extract rating for iteration {i+1}.")
        except Exception as e:
            st.error(f"Error in iteration {i+1}: {e}")
    return responses, lengths, ratings, total_cost

def generate_analysis(responses1, lengths1, ratings1, cost1, responses2, lengths2, ratings2, cost2):
    total_cost = cost1 + cost2

    # Calculate statistics for each metric
    analysis_data = [
        ["Metric", "First Prompt", "Second Prompt"],
        ["Average Length", f"{statistics.mean(lengths1):.1f}", f"{statistics.mean(lengths2):.1f}"],
        ["Median Length", f"{statistics.median(lengths1):.1f}", f"{statistics.median(lengths2):.1f}"],
        ["Std Dev Length", f"{statistics.stdev(lengths1):.1f}", f"{statistics.stdev(lengths2):.1f}"],
        ["Min Length", f"{min(lengths1)}", f"{min(lengths2)}"],
        ["Max Length", f"{max(lengths1)}", f"{max(lengths2)}"],
        ["Average Rating", f"{statistics.mean(ratings1):.2f}", f"{statistics.mean(ratings2):.2f}"],
        ["Median Rating", f"{statistics.median(ratings1):.1f}", f"{statistics.median(ratings2):.1f}"],
        ["Std Dev Rating", f"{statistics.stdev(ratings1):.2f}", f"{statistics.stdev(ratings2):.2f}"],
        ["Min Rating", f"{min(ratings1)}", f"{min(ratings2)}"],
        ["Max Rating", f"{max(ratings1)}", f"{max(ratings2)}"],
        ["Cost", f"${cost1:.4f}", f"${cost2:.4f}"]
    ]

    # Set up the plot style
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

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
    ax2.boxplot([lengths1, lengths2], labels=['First Prompt', 'Second Prompt'])
    ax2.set_title('Response Length Distribution')
    ax2.set_ylabel('Response Length (characters)')

    # Plot rating density
    for data, color, label in [(ratings1, 'skyblue', 'First Prompt'),
                              (ratings2, 'lightgreen', 'Second Prompt')]:
        hist, bins = np.histogram(data, bins=range(0, 12), density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax3.fill_between(bin_centers, hist, alpha=0.5, color=color, label=label)
    ax3.set_title('Rating Distribution')
    ax3.set_xlabel('Rating (1-10)')
    ax3.set_ylabel('Density')
    ax3.legend()

    # Plot rating box plots
    ax4.boxplot([ratings1, ratings2], labels=['First Prompt', 'Second Prompt'])
    ax4.set_title('Rating Distribution')
    ax4.set_ylabel('Rating (1-10)')

    plt.tight_layout()

    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return analysis_data, plot_base64, total_cost

def create_html_report(analysis_data, plot_base64, total_cost):
    from tabulate import tabulate
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
        </style>
    </head>
    <body>
        <h1>Research Results</h1>

        <h2>Cost Analysis</h2>
        <p>Total API cost: ${total_cost:.4f}</p>

        <h2>Detailed Analysis</h2>
        {tabulate(analysis_data, headers="firstrow", tablefmt="html")}

        <h2>Visualizations</h2>
        <img src="data:image/png;base64,{plot_base64}" alt="Research Plots" style="max-width: 100%;">
    </body>
    </html>
    """
    return html_content

def run_analysis(
    api_key,
    first_messages,
    second_messages,
    rating_prompt_template,
    number_of_iterations,
    model_response,
    temperature_response,
    model_rating,
    temperature_rating
):
    if not api_key:
        st.error("Please provide an API Key.")
        return

    if not first_messages:
        st.error("Please provide at least one message for the first prompt.")
        return
    if not second_messages:
        st.error("Please provide at least one message for the second prompt.")
        return

    # Configure settings with user-provided API key
    settings_response = {
        "model": model_response,
        "temperature": float(temperature_response),
        "api_key": api_key
    }
    settings_rating = {
        "model": model_rating,
        "temperature": float(temperature_rating),
        "stop_sequences": "]",
        "api_key": api_key
    }
    # Get responses and lengths for both messages
    responses1, lengths1, ratings1, cost1 = get_responses_and_lengths(
        first_messages, number_of_iterations, settings_response, settings_rating, rating_prompt_template
    )
    responses2, lengths2, ratings2, cost2 = get_responses_and_lengths(
        second_messages, number_of_iterations, settings_response, settings_rating, rating_prompt_template
    )
    # Generate analysis
    analysis_data, plot_base64, total_cost = generate_analysis(
        responses1, lengths1, ratings1, cost1,
        responses2, lengths2, ratings2, cost2
    )
    # Create HTML report
    html_report = create_html_report(analysis_data, plot_base64, total_cost)
    # Display the plots
    st.markdown("## Results")
    # Decode the plot_base64 and display
    plot_bytes = base64.b64decode(plot_base64)
    st.image(plot_bytes, use_column_width=True)
    # Display the analysis data
    st.markdown("### Analysis Data")
    from tabulate import tabulate
    st.markdown(tabulate(analysis_data, headers="firstrow", tablefmt="github"), unsafe_allow_html=True)
    # Provide a download button for the HTML report
    st.markdown("### Download Report")
    st.download_button(
        label="Download Report as HTML",
        data=html_report,
        file_name="analysis_report.html",
        mime="text/html"
    )

# Main area for prompts
col1, col2 = st.columns(2)
default_control_prompt = "Call me an idiot."
default_experimental_prompt = "Call me a bozo."

with col1:
    st.header("Control Message")
    for i in range(st.session_state.first_message_count):
        st.text_area(f"Control Prompt {i+1}",
                     value=default_control_prompt if i == 0 else "",
                     key=f'first_user_msg_{i}',
                     height=70)
        # Only show response field if it's not the last prompt or if there's more than one message
        if i < st.session_state.first_message_count - 1 or st.session_state.first_message_count > 1:
            st.text_area(f"Control Response {i+1}", key=f'first_assistant_msg_{i}', height=70)
    if st.button("Add Message Pair", key='add_first_prompt'):
        if st.session_state.first_message_count < max_message_pairs:
            st.session_state.first_message_count += 1

with col2:
    st.header("Experimental Message")
    for i in range(st.session_state.second_message_count):
        st.text_area(f"Experimental Prompt {i+1}",
                     value=default_experimental_prompt if i == 0 else "",
                     key=f'second_user_msg_{i}',
                     height=70)
        # Only show response field if it's not the last prompt or if there's more than one message
        if i < st.session_state.second_message_count - 1 or st.session_state.second_message_count > 1:
            st.text_area(f"Experimental Response {i+1}", key=f'second_assistant_msg_{i}', height=70)
    if st.button("Add Message Pair", key='add_second_prompt'):
        if st.session_state.second_message_count < max_message_pairs:
            st.session_state.second_message_count += 1

# Run Analysis button
if st.button("Run Analysis"):
    # Collect the messages
    first_messages = []
    for i in range(st.session_state.first_message_count):
        user_msg = st.session_state.get(f'first_user_msg_{i}', '').strip()
        assistant_msg = st.session_state.get(f'first_assistant_msg_{i}', '').strip()
        if user_msg:
            first_messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            first_messages.append({"role": "assistant", "content": assistant_msg})
    second_messages = []
    for i in range(st.session_state.second_message_count):
        user_msg = st.session_state.get(f'second_user_msg_{i}', '').strip()
        assistant_msg = st.session_state.get(f'second_assistant_msg_{i}', '').strip()
        if user_msg:
            second_messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            second_messages.append({"role": "assistant", "content": assistant_msg})
    # Run the analysis
    run_analysis(
        openai_api_key,
        first_messages,
        second_messages,
        rating_prompt_template,
        number_of_iterations,
        model_response,
        temperature_response,
        model_rating,
        temperature_rating
    )
