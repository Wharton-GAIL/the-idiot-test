import streamlit as st
import statistics
import re
import base64
import io
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from call_gpt import call_gpt

def generate_analysis(
    responses_ctrl, lengths_ctrl, ratings_ctrl, cost_ctrl,
    responses_exp, lengths_exp, ratings_exp, cost_exp,
    analyze_length=True
):
    total_cost = cost_ctrl + cost_exp

    analysis_data = [["Metric", "Control Prompt", "Experimental Prompt"]]

    if analyze_length:
        analysis_data.extend([
            ["Average Length", f"{statistics.mean(lengths_ctrl):.1f}", f"{statistics.mean(lengths_exp):.1f}"],
            ["Median Length", f"{statistics.median(lengths_ctrl):.1f}", f"{statistics.median(lengths_exp):.1f}"],
            ["Std Dev Length", f"{statistics.stdev(lengths_ctrl):.1f}", f"{statistics.stdev(lengths_exp):.1f}"],
            ["Min Length", f"{min(lengths_ctrl)}", f"{min(lengths_exp)}"],
            ["Max Length", f"{max(lengths_ctrl)}", f"{max(lengths_exp)}"],
        ])

    analysis_data.extend([
        ["Average Rating", f"{statistics.mean(ratings_ctrl):.2f}", f"{statistics.mean(ratings_exp):.2f}"],
        ["Median Rating", f"{statistics.median(ratings_ctrl):.1f}", f"{statistics.median(ratings_exp):.1f}"],
        ["Std Dev Rating", f"{statistics.stdev(ratings_ctrl):.2f}", f"{statistics.stdev(ratings_exp):.2f}"],
        ["Min Rating", f"{min(ratings_ctrl)}", f"{min(ratings_exp)}"],
        ["Max Rating", f"{max(ratings_ctrl)}", f"{max(ratings_exp)}"],
        ["Cost", f"${cost_ctrl:.4f}", f"${cost_exp:.4f}"]
    ])

    plt.style.use('default')

    if analyze_length:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    else:
        fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))

    if analyze_length:
        for data, color, label in [(lengths_ctrl, 'skyblue', 'Control Prompt'),
                                   (lengths_exp, 'lightgreen', 'Experimental Prompt')]:
            hist, bins = np.histogram(data, bins=10, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax1.fill_between(bin_centers, hist, alpha=0.5, color=color, label=label)
        ax1.set_title('Response Length Distribution')
        ax1.set_xlabel('Response Length (characters)')
        ax1.set_ylabel('Density')
        ax1.legend()

        ax2.boxplot([lengths_ctrl, lengths_exp], labels=['Control Prompt', 'Experimental Prompt'])
        ax2.set_title('Response Length Box Plot')
        ax2.set_ylabel('Response Length (characters)')

    for data, color, label in [(ratings_ctrl, 'skyblue', 'Control Prompt'),
                               (ratings_exp, 'lightgreen', 'Experimental Prompt')]:
        max_rating = max(max(ratings_ctrl), max(ratings_exp))
        hist, bins = np.histogram(data, bins=range(0, int(max_rating) + 2), density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax3.fill_between(bin_centers, hist, alpha=0.5, color=color, label=label)
    ax3.set_title('Rating Distribution')
    ax3.set_xlabel('Rating')
    ax3.set_ylabel('Density')
    ax3.legend()

    ax4.boxplot([ratings_ctrl, ratings_exp], labels=['Control Prompt', 'Experimental Prompt'])
    ax4.set_title('Rating Box Plot')
    ax4.set_ylabel('Rating')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return analysis_data, plot_base64, total_cost

def construct_conversation_html(prompts, responses = None):
    conversation = []
    print("Prompts: "+str(prompts)+"\n\n")
    print("Responses: "+str(responses)+"\n\n")
    if prompts:
        for msg in prompts:
            emoji = '🧑' if msg['role'] == 'user' else '🤖'
            conversation.append(f"{emoji} {msg['content']}")
    if responses:
        for user_prompt, assistant_response in zip(prompts, responses):
            conversation.append(f"🧑 {user_prompt}")
            conversation.append(f"🤖 {assistant_response}")
    
    # Join with double line breaks for readability
    conversation_html = ('<br>').join(conversation)
    return conversation_html

def create_html_report(
    analysis_data, plot_base64, total_cost, messages_ctrl, messages_exp,
    control_rating_prompt_template, experimental_rating_prompt_template,
    show_raw_results=False, responses_ctrl=None, responses_exp=None,
    ratings_ctrl=None, ratings_exp=None, rating_texts_ctrl=None, rating_texts_exp=None,
    model_response=None, model_rating=None, temperature_response=None, temperature_rating=None
):
    # Ensure that iteration prompts and responses are lists
    responses_ctrl = responses_ctrl or []
    responses_exp = responses_exp or []
    
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
            .config-columns {{
                display: flex;
                gap: 20px;
            }}
            .config-column {{
                flex: 1;
            }}
            .prompt-container {{
                margin: 20px 0;
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
            <p><strong>Number of Iterations:</strong> {len(responses_ctrl)}</p>
            <p><strong>Response Generation Model:</strong> {model_response}</p>
            <p><strong>Response Temperature:</strong> {temperature_response}</p>
            <p><strong>Rating Model:</strong> {model_rating}</p>
            <p><strong>Rating Temperature:</strong> {temperature_rating}</p>
            <div class="config-columns">
                <div class="config-column">
                    <h3>Control Messages</h3>
                    <div class="response-box">
                        {construct_conversation_html(prompts=messages_ctrl)}
                    </div>
                    <h4>Control Rating Prompt</h4>
                    <div class="response-box">🧑 {control_rating_prompt_template}</div>
                </div>
                <div class="config-column">
                    <h3>Experimental Messages</h3>
                    <div class="response-box">
                        {construct_conversation_html(prompts=messages_exp)}
                    </div>
                    <h4>Experimental Rating Prompt</h4>
                    <div class="response-box">🧑 {experimental_rating_prompt_template}</div>
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

    if show_raw_results and all([responses_ctrl, responses_exp, ratings_ctrl, ratings_exp]):
        html_content += """
        <h2>Raw Results</h2>
        """

        num_iterations = len(responses_ctrl)

        for idx in range(num_iterations):
            html_content += f"""
            <h3>Iteration {idx + 1}</h3>
            """

            html_content += """
            <h4>Control Reply</h4>
            """

            if idx < len(responses_ctrl):
                html_content += f"""
                <div class="response-box">🤖 {responses_ctrl[idx]}</div>
                """
            else:
                html_response = "N/A"
                html_content += f"""
                <div class="response-box">🤖 {html_response}</div>
                """

            if ratings_ctrl and idx < len(ratings_ctrl) and rating_texts_ctrl and idx < len(rating_texts_ctrl):
                html_content += f"""
                <h4>Control Rating Text (Rating: {ratings_ctrl[idx]})</h4>
                <div class="response-box">🤖 {rating_texts_ctrl[idx]}</div>
                """
            else:
                html_content += f"""
                <h4>Control Rating Text</h4>
                <div class="response-box">🤖 N/A</div>
                """

            html_content += """
            <h4>Experimental Reply</h4>
            """

            if idx < len(responses_exp):
                html_content += f"""
                <div class="response-box">🤖 {responses_exp[idx]}</div>
                """
            else:
                html_response = "N/A"
                html_content += f"""
                <div class="response-box">🤖 {html_response}</div>
                """

            if ratings_exp and idx < len(ratings_exp) and rating_texts_exp and idx < len(rating_texts_exp):
                html_content += f"""
                <h4>Experimental Rating Text (Rating: {ratings_exp[idx]})</h4>
                <div class="response-box">🤖 {rating_texts_exp[idx]}</div>
                """
            else:
                html_content += f"""
                <h4>Experimental Rating Text</h4>
                <div class="response-box">🤖 N/A</div>
                """

    html_content += """
    </body>
    </html>
    """

    return html_content
