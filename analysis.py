import streamlit as st
import statistics
import re
import base64
import io
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from call_gpt import call_gpt

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

def generate_analysis(
    responses1, lengths1, ratings1, cost1,
    responses2, lengths2, ratings2, cost2,
    analyze_length=True
):
    total_cost = cost1 + cost2

    analysis_data = [["Metric", "First Prompt", "Second Prompt"]]

    if analyze_length:
        analysis_data.extend([
            ["Average Length", f"{statistics.mean(lengths1):.1f}", f"{statistics.mean(lengths2):.1f}"],
            ["Median Length", f"{statistics.median(lengths1):.1f}", f"{statistics.median(lengths2):.1f}"],
            ["Std Dev Length", f"{statistics.stdev(lengths1):.1f}", f"{statistics.stdev(lengths2):.1f}"],
            ["Min Length", f"{min(lengths1)}", f"{min(lengths2)}"],
            ["Max Length", f"{max(lengths1)}", f"{max(lengths2)}"],
        ])

    analysis_data.extend([
        ["Average Rating", f"{statistics.mean(ratings1):.2f}", f"{statistics.mean(ratings2):.2f}"],
        ["Median Rating", f"{statistics.median(ratings1):.1f}", f"{statistics.median(ratings2):.1f}"],
        ["Std Dev Rating", f"{statistics.stdev(ratings1):.2f}", f"{statistics.stdev(ratings2):.2f}"],
        ["Min Rating", f"{min(ratings1)}", f"{min(ratings2)}"],
        ["Max Rating", f"{max(ratings1)}", f"{max(ratings2)}"],
        ["Cost", f"${cost1:.4f}", f"${cost2:.4f}"]
    ])

    plt.style.use('default')

    if analyze_length:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    else:
        fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))

    if analyze_length:
        for data, color, label in [(lengths1, 'skyblue', 'First Prompt'),
                                   (lengths2, 'lightgreen', 'Second Prompt')]:
            hist, bins = np.histogram(data, bins=10, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax1.fill_between(bin_centers, hist, alpha=0.5, color=color, label=label)
        ax1.set_title('Response Length Distribution')
        ax1.set_xlabel('Response Length (characters)')
        ax1.set_ylabel('Density')
        ax1.legend()

        ax2.boxplot([lengths1, lengths2], labels=['First Prompt', 'Second Prompt'])
        ax2.set_title('Response Length Box Plot')
        ax2.set_ylabel('Response Length (characters)')

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

    ax4.boxplot([ratings1, ratings2], labels=['First Prompt', 'Second Prompt'])
    ax4.set_title('Rating Box Plot')
    ax4.set_ylabel('Rating')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return analysis_data, plot_base64, total_cost

def create_html_report(
    analysis_data, plot_base64, total_cost, first_messages, second_messages,
    control_rating_prompt_template, experimental_rating_prompt_template,
    show_raw_results=False, responses1=None, responses2=None,
    ratings1=None, ratings2=None, rating_texts1=None, rating_texts2=None
):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Results</title>
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
                        {'<br>'.join(msg['content'] for msg in first_messages)}
                    </div>

                    <h4>Control Rating Prompt</h4>
                    <div class="response-box">{control_rating_prompt_template}</div>
                </div>

                <div class="prompt-section">
                    <h3>Experimental Messages</h3>
                    <div class="response-box">
                        {'<br>'.join(msg['content'] for msg in second_messages)}
                    </div>

                    <h4>Experimental Rating Prompt</h4>
                    <div class="response-box">{experimental_rating_prompt_template}</div>
                </div>
            </div>
        </div>

        <h2>Cost Analysis</h2>
        <p>Total API cost: ${total_cost:.4f}</p>

        <h2>Detailed Analysis</h2>
        {tabulate(analysis_data, headers='firstrow', tablefmt='html')}

        <h2>Visualizations</h2>
        <img src="data:image/png;base64,{plot_base64}" alt="Research Plots" style="max-width: 100%;">
    """

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
