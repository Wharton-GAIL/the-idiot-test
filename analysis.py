import streamlit as st
import statistics
import base64
import io
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

def generate_analysis(
    responses_ctrl, lengths_ctrl, ratings_ctrl, cost_ctrl,
    responses_exp, lengths_exp, ratings_exp, cost_exp,
    analyze_rating=True,
    analyze_length=True
):
    total_cost = cost_ctrl + cost_exp

    analysis_data = [["Metric", "Control Prompt", "Experimental Prompt"]]

    if analyze_length and lengths_ctrl and lengths_exp:
        analysis_data.extend([
            ["Average Length", f"{statistics.mean(lengths_ctrl):.1f}", f"{statistics.mean(lengths_exp):.1f}"],
            ["Median Length", f"{statistics.median(lengths_ctrl):.1f}", f"{statistics.median(lengths_exp):.1f}"],
            ["Std Dev Length", f"{statistics.stdev(lengths_ctrl):.1f}" if len(lengths_ctrl) > 1 else "N/A",
             f"{statistics.stdev(lengths_exp):.1f}" if len(lengths_exp) > 1 else "N/A"],
            ["Min Length", f"{min(lengths_ctrl)}", f"{min(lengths_exp)}"],
            ["Max Length", f"{max(lengths_ctrl)}", f"{max(lengths_exp)}"],
        ])

    if analyze_rating and ratings_ctrl and ratings_exp:
        analysis_data.extend([
            ["Average Rating", f"{statistics.mean(ratings_ctrl):.2f}", f"{statistics.mean(ratings_exp):.2f}"],
            ["Median Rating", f"{statistics.median(ratings_ctrl):.1f}", f"{statistics.median(ratings_exp):.1f}"],
            ["Std Dev Rating", f"{statistics.stdev(ratings_ctrl):.2f}" if len(ratings_ctrl) > 1 else "N/A",
             f"{statistics.stdev(ratings_exp):.2f}" if len(ratings_exp) > 1 else "N/A"],
            ["Min Rating", f"{min(ratings_ctrl)}", f"{min(ratings_exp)}"],
            ["Max Rating", f"{max(ratings_ctrl)}", f"{max(ratings_exp)}"],
        ])

    analysis_data.append(["Cost", f"${cost_ctrl:.4f}", f"${cost_exp:.4f}"])

    plt.style.use('default')

    # Determine subplot layout
    plot_rows = 1
    plot_cols = 0
    if analyze_length and lengths_ctrl and lengths_exp:
        plot_cols += 2
    if analyze_rating and ratings_ctrl and ratings_exp:
        plot_cols += 2

    if plot_cols == 0:
        # No plots to generate
        plot_base64 = ""
    else:
        fig, axs = plt.subplots(plot_rows, plot_cols, figsize=(5 * plot_cols, 6))
        if plot_cols == 1:
            axs = [axs]
        else:
            axs = axs.flatten()

        plot_idx = 0

        if analyze_length and lengths_ctrl and lengths_exp:
            for data, color, label in [(lengths_ctrl, 'skyblue', 'Control Prompt'),
                                       (lengths_exp, 'lightgreen', 'Experimental Prompt')]:
                hist, bins = np.histogram(data, bins=10, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                axs[plot_idx].fill_between(bin_centers, hist, alpha=0.5, color=color, label=label)
            axs[plot_idx].set_title('Response Length Distribution')
            axs[plot_idx].set_xlabel('Response Length (characters)')
            axs[plot_idx].set_ylabel('Density')
            axs[plot_idx].legend()
            plot_idx += 1

            axs[plot_idx].boxplot([lengths_ctrl, lengths_exp], labels=['Control Prompt', 'Experimental Prompt'])
            axs[plot_idx].set_title('Response Length Box Plot')
            axs[plot_idx].set_ylabel('Response Length (characters)')
            plot_idx += 1

        if analyze_rating and ratings_ctrl and ratings_exp:
            max_rating = max(max(ratings_ctrl), max(ratings_exp))
            bins = np.arange(0, int(max_rating) + 2)  # +2 to include the max value
            
            # Calculate histograms
            hist_ctrl, _ = np.histogram(ratings_ctrl, bins=bins)
            hist_exp, _ = np.histogram(ratings_exp, bins=bins)
            
            # Normalize to get proportions
            hist_ctrl = hist_ctrl / len(ratings_ctrl)
            hist_exp = hist_exp / len(ratings_exp)
            
            # Plot bars side by side
            bar_width = 0.35
            x = np.arange(len(bins)-1)
            axs[plot_idx].bar(x - bar_width/2, hist_ctrl, bar_width, label='Control Prompt', color='skyblue', alpha=0.7)
            axs[plot_idx].bar(x + bar_width/2, hist_exp, bar_width, label='Experimental Prompt', color='lightgreen', alpha=0.7)
            
            axs[plot_idx].set_title('Rating Distribution')
            axs[plot_idx].set_xlabel('Rating')
            axs[plot_idx].set_ylabel('Proportion')
            axs[plot_idx].set_xticks(x)
            axs[plot_idx].set_xticklabels([f'{i:.1f}' for i in bins[:-1]])
            axs[plot_idx].legend()
            plot_idx += 1

            axs[plot_idx].boxplot([ratings_ctrl, ratings_exp], labels=['Control Prompt', 'Experimental Prompt'])
            axs[plot_idx].set_title('Rating Box Plot')
            axs[plot_idx].set_ylabel('Rating')
            plot_idx += 1

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

    return analysis_data, plot_base64, total_cost

def create_html_report(
    analysis_data,
    plot_base64,
    total_cost,
    messages_ctrl_original,
    messages_exp_original,
    messages_ctrl_modified,
    messages_exp_modified,
    control_rating_prompt_template,
    experimental_rating_prompt_template,
    analyze_rating=True,
    show_transcripts=False,
    responses_ctrl=None,
    responses_exp=None,
    ratings_ctrl=None,
    ratings_exp=None,
    rating_texts_ctrl=None,
    rating_texts_exp=None,
    model_response=None,
    model_rating=None,
    temperature_response=None,
    temperature_rating=None,
    control_system_message=None,
    experiment_system_message=None,
):
    # Construct HTML for original prompts
    def extract_user_messages(messages):
        conversation = []
        for msg in messages:
            if msg['role'] == 'user':
                conversation.append(f"&#x1F9D1; {msg['content']}")
            elif msg['role'] == 'assistant':
                # Show [AI Responds] for blank assistant messages
                content = msg['content'].strip()
                conversation.append(f"&#x1F916; {content if content else '[AI Responds]'}")
        conversation_html = ('<br>').join(conversation)
        return conversation_html

    system_messages_html = f"""
    <div class="config-section" style="margin-bottom: 0;">
        <div class="config-columns">
            <div class="config-column">
                <div class="prompt-label">Control System Message:</div>
                <div class="response-box" style="margin-bottom: 0;">{control_system_message if control_system_message else 'None'}</div>
            </div>
            <div class="config-column">
                <div class="prompt-label">Experimental System Message:</div>
                <div class="response-box">{experiment_system_message if experiment_system_message else 'None'}</div>
            </div>
        </div>
    </div>
    """

    original_prompts_html = f"""
    <div class="config-section">
        <div class="config-columns">
            <div class="config-column">
                <div class="prompt-label" style="margin-top: 0;">Control Prompts</div>
                <div class="response-box">{extract_user_messages(messages_ctrl_original)}</div>
            </div>
            <div class="config-column">
                <div class="prompt-label">Experimental Prompts</div>
                <div class="response-box">{extract_user_messages(messages_exp_original)}</div>
            </div>
        </div>
    </div>
    """

    # CSS styles with escaped curly braces
    css_styles = """
        body { font-family: Arial, sans-serif; margin: 40px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .response-box {
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .info-section {
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .config-columns {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .config-column {
            flex: 1;
            min-width: 250px;
        }
        .prompt-container {
            margin: 20px 0;
        }
        .prompt-label {
            font-weight: bold;
            color: #555;
            margin-bottom: 5px;
            display: block;
        }
        .config-section {
            margin-bottom: 40px;
        }
        h1, h2, h3, h4 {
            color: #333;
        }
    """

    # Build the HTML content using f-strings
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Results</title>
        <style>
            {css_styles}
        </style>
    </head>
    <body>
        <h1>Research Results</h1>

        <div class="info-section">
            <h2>Configuration</h2>
            <p><strong>Number of Iterations:</strong> {len(responses_ctrl) if responses_ctrl else 'N/A'}</p>
            <p><strong>Response Generation Model:</strong> {model_response if model_response else 'N/A'}</p>
            <p><strong>Response Temperature:</strong> {temperature_response if temperature_response else 'N/A'}</p>
            {f"<p><strong>Rating Model:</strong> {model_rating}</p>" if analyze_rating and model_rating else ""}
            {f"<p><strong>Rating Temperature:</strong> {temperature_rating}</p>" if analyze_rating and temperature_rating else ""}
            {system_messages_html}
            {original_prompts_html}
        </div>

        <h2>Cost Analysis</h2>
        <p>Total API cost: ${total_cost:.4f}</p>

        <h2>Detailed Analysis</h2>
        {tabulate(analysis_data, headers='firstrow', tablefmt='html')}

        {f"<h2>Visualizations</h2><img src='data:image/png;base64,{plot_base64}' alt='Research Plots' style='max-width: 100%;'>" if plot_base64 else ""}
    """

    if show_transcripts and responses_ctrl and responses_exp:
        transcripts_html = "<h2>Transcripts</h2>"
        num_iterations = len(responses_ctrl)

        for idx in range(num_iterations):
            iteration_html = f"""
            <h3>Iteration {idx + 1}</h3>
            """

            # Control conversation
            control_conv_html = "<h4>Control Conversation</h4>"
            for msg in messages_ctrl_modified[idx]:
                if msg['role'] == 'user':
                    control_conv_html += f"""
                    <div class="response-box">&#x1F9D1; {msg['content']}</div>
                    """
                elif msg['role'] == 'assistant':
                    content = msg['content'].strip() if msg['content'].strip() else '[AI Responds]'
                    control_conv_html += f"""
                    <div class="response-box">&#x1F916; {content}</div>
                    """

            if analyze_rating:
                if ratings_ctrl and idx < len(ratings_ctrl) and rating_texts_ctrl and idx < len(rating_texts_ctrl):
                    control_conv_html += f"""
                    <h4>Control Rating Text (Rating: {ratings_ctrl[idx]})</h4>
                    <div class="response-box">&#x1F916; {rating_texts_ctrl[idx]}</div>
                    """
                else:
                    control_conv_html += """
                    <h4>Control Rating Text</h4>
                    <div class="response-box">&#x1F916; N/A</div>
                    """

            # Experimental conversation
            exp_conv_html = "<h4>Experimental Conversation</h4>"
            for msg in messages_exp_modified[idx]:
                if msg['role'] == 'user':
                    exp_conv_html += f"""
                    <div class="response-box">&#x1F9D1; {msg['content']}</div>
                    """
                elif msg['role'] == 'assistant':
                    content = msg['content'].strip() if msg['content'].strip() else '[AI Responds]'
                    exp_conv_html += f"""
                    <div class="response-box">&#x1F916; {content}</div>
                    """

            if analyze_rating:
                if ratings_exp and idx < len(ratings_exp) and rating_texts_exp and idx < len(rating_texts_exp):
                    exp_conv_html += f"""
                    <h4>Experimental Rating Text (Rating: {ratings_exp[idx]})</h4>
                    <div class="response-box">&#x1F916; {rating_texts_exp[idx]}</div>
                    """
                else:
                    exp_conv_html += """
                    <h4>Experimental Rating Text</h4>
                    <div class="response-box">&#x1F916; N/A</div>
                    """

            # Combine iteration HTML
            transcripts_html += iteration_html + control_conv_html + exp_conv_html

        html_content += transcripts_html

    # Close the HTML tags
    html_content += """
    </body>
    </html>
    """

    return html_content
