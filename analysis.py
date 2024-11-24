import streamlit as st
import statistics
import base64
import io
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

def generate_analysis(chat_results, analyze_rating=True, analyze_length=True):
    total_cost = sum(chat_res["total_cost"] for chat_res in chat_results.values())

    # Prepare the analysis data table
    headers = ["Metric"] + [f"Chat {chat_index}" for chat_index in sorted(chat_results.keys())]
    analysis_data = [headers]

    if analyze_length:
        lengths_data = []
        for chat_index in sorted(chat_results.keys()):
            lengths = chat_results[chat_index]["lengths"]
            lengths_data.append(lengths)

        metrics = ["Average Length", "Median Length", "Std Dev Length", "Min Length", "Max Length"]
        stats_funcs = [statistics.mean, statistics.median, statistics.stdev, min, max]
        for metric, func in zip(metrics, stats_funcs):
            row = [metric]
            for lengths in lengths_data:
                if lengths:
                    if func == statistics.stdev and len(lengths) < 2:
                        value = "N/A"
                    else:
                        value = f"{func(lengths):.1f}"
                else:
                    value = "N/A"
                row.append(value)
            analysis_data.append(row)

    if analyze_rating:
        ratings_data = []
        for chat_index in sorted(chat_results.keys()):
            ratings = chat_results[chat_index]["ratings"]
            ratings_data.append(ratings)

        metrics = ["Average Rating", "Median Rating", "Std Dev Rating", "Min Rating", "Max Rating"]
        stats_funcs = [statistics.mean, statistics.median, statistics.stdev, min, max]
        for metric, func in zip(metrics, stats_funcs):
            row = [metric]
            for ratings in ratings_data:
                if ratings:
                    if func == statistics.stdev and len(ratings) < 2:
                        value = "N/A"
                    else:
                        value = f"{func(ratings):.2f}"
                else:
                    value = "N/A"
                row.append(value)
            analysis_data.append(row)

    # Add total cost per chat
    row = ["Cost"]
    for chat_index in sorted(chat_results.keys()):
        cost = chat_results[chat_index]["total_cost"]
        row.append(f"${cost:.4f}")
    analysis_data.append(row)

    # Generate plots
    plot_base64 = ""
    plt.style.use('default')

    plot_cols = 0
    if analyze_length:
        plot_cols += 2
    if analyze_rating:
        plot_cols += 2

    if plot_cols == 0:
        # No plots to generate
        plot_base64 = ""
    else:
        fig, axs = plt.subplots(1, plot_cols, figsize=(5 * plot_cols, 6))
        if plot_cols == 1:
            axs = [axs]
        else:
            axs = axs.flatten()

        plot_idx = 0

        if analyze_length:
            # Length distribution histogram
            for chat_index in sorted(chat_results.keys()):
                lengths = chat_results[chat_index]["lengths"]
                hist, bins = np.histogram(lengths, bins=10, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                axs[plot_idx].fill_between(bin_centers, hist, alpha=0.5, label=f"Chat {chat_index}")
            axs[plot_idx].set_title('Response Length Distribution')
            axs[plot_idx].set_xlabel('Response Length (characters)')
            axs[plot_idx].set_ylabel('Density')
            axs[plot_idx].legend()
            plot_idx += 1

            # Length boxplot
            lengths_data = [chat_results[chat_index]["lengths"] for chat_index in sorted(chat_results.keys())]
            labels = [f"Chat {chat_index}" for chat_index in sorted(chat_results.keys())]
            axs[plot_idx].boxplot(lengths_data, labels=labels)
            axs[plot_idx].set_title('Response Length Box Plot')
            axs[plot_idx].set_ylabel('Response Length (characters)')
            plot_idx += 1

        if analyze_rating:
            # Rating distribution histogram
            max_rating = max(
                max(chat_results[chat_index]["ratings"]) for chat_index in chat_results if chat_results[chat_index]["ratings"]
            )
            bins = np.arange(0, int(max_rating) + 2)  # +2 to include the max value

            for idx, chat_index in enumerate(sorted(chat_results.keys())):
                ratings = chat_results[chat_index]["ratings"]
                hist, _ = np.histogram(ratings, bins=bins)
                # Normalize to get proportions
                hist = hist / len(ratings)
                x = np.arange(len(bins)-1)
                bar_width = 0.8 / len(chat_results)
                axs[plot_idx].bar(x + (idx * bar_width), hist, bar_width,
                                  label=f"Chat {chat_index}", alpha=0.7)
            axs[plot_idx].set_title('Rating Distribution')
            axs[plot_idx].set_xlabel('Rating')
            axs[plot_idx].set_ylabel('Proportion')
            axs[plot_idx].set_xticks(x)
            axs[plot_idx].set_xticklabels([f'{i:.1f}' for i in bins[:-1]])
            axs[plot_idx].legend()
            plot_idx += 1

            # Rating boxplot
            ratings_data = [chat_results[chat_index]["ratings"] for chat_index in sorted(chat_results.keys())]
            labels = [f"Chat {chat_index}" for chat_index in sorted(chat_results.keys())]
            axs[plot_idx].boxplot(ratings_data, labels=labels)
            axs[plot_idx].set_title('Rating Box Plot')
            axs[plot_idx].set_ylabel('Rating')
            plot_idx += 1

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

    # Return analysis data, plot, and total cost
    return analysis_data, plot_base64, total_cost

def create_html_report(
    analysis_data,
    plot_base64,
    total_cost,
    chat_data,
    chat_results,
    model_response=None,
    model_rating=None,
    temperature_response=None,
    temperature_rating=None,
    evaluation_rubrics=None,
    analyze_rating=True,
    show_transcripts=False,
):
    # Construct combined HTML for each chat's system message, prompts, and evaluation rubric
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

    combined_html = ""

    for idx, chat_info in enumerate(chat_data, start=1):
        rubric_html = ""
        if analyze_rating and evaluation_rubrics and idx in evaluation_rubrics:
            rubric = evaluation_rubrics[idx]
            rubric_html = f"""
                <div class='response-box'>
                    {rubric}
                </div>
            """
        
        combined_html += f"""
        <div class="config-section" style="margin-bottom: 0;">
            <p><strong>Chat {idx}</strong></p>
            <div class="config-columns">
                <div class="config-column">
                    <div class="prompt-label">System Message:</div>
                    <div class="response-box" style="margin-bottom: 0;">{chat_info['system_message'] if chat_info['system_message'] else 'None'}</div>
                </div>
                <div class="config-column">
                    <div class="prompt-label" style="margin-top: 0;">Prompts:</div>
                    <div class="response-box">{extract_user_messages([msg for msg in chat_info['messages'] if msg['role'] == 'user' or msg['role'] == 'assistant'])}</div>
                </div>
            </div>
                <div class="prompt-label">Evaluation Rubric:</div>
                {rubric_html}
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
        h1, h2, h3, h4, h5 {
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
            <p><strong>Number of Iterations:</strong> {len(next(iter(chat_results.values()))['responses']) if chat_results else 'N/A'}</p>
            <p><strong>Response Generation Model:</strong> {model_response if model_response else 'N/A'}</p>
            <p><strong>Response Temperature:</strong> {temperature_response if temperature_response else 'N/A'}</p>
            {f"<p><strong>Rating Model:</strong> {model_rating}</p>" if analyze_rating and model_rating else ""}
            {f"<p><strong>Rating Temperature:</strong> {temperature_rating}</p>" if analyze_rating and temperature_rating else ""}
            {f"<div class='response-box'><strong>Evaluation Rubric:</strong><br>{evaluation_rubric}</div>" if False else ""}  <!-- Removed global rubric display -->
            {combined_html}
        </div>

        <h2>Cost Analysis</h2>
        <p>Total API cost: ${total_cost:.4f}</p>

        <h2>Detailed Analysis</h2>
        {tabulate(analysis_data, headers='firstrow', tablefmt='html')}

        {f"<h2>Visualizations</h2><img src='data:image/png;base64,{plot_base64}' alt='Research Plots' style='max-width: 100%;'>" if plot_base64 else ""}
    """

    if show_transcripts and chat_results:
        transcripts_html = "<h2>Transcripts</h2>"

        # Sort chat_results by chat index to ensure Chat 1, Chat 2, etc.
        for chat_index in sorted(chat_results.keys()):
            results = chat_results[chat_index]
            transcripts_html += f"<h3>Chat {chat_index}</h3>"

            # Sort iterations to ensure Iteration 1, Iteration 2, etc.
            for iteration_idx, messages in enumerate(results["messages_per_iteration"], start=1):
                transcripts_html += f"<h4>Iteration {iteration_idx}</h4>"
                conversation_html = "<div class='response-box'>"
                for msg in messages:
                    if msg['role'] == 'user':
                        conversation_html += f"&#x1F9D1; {msg['content']}<br>"
                    elif msg['role'] == 'assistant':
                        content = msg['content'].strip() if msg['content'].strip() else '[AI Responds]'
                        conversation_html += f"&#x1F916; {content}<br>"
                conversation_html += "</div>"
                transcripts_html += conversation_html

                if analyze_rating:
                    rating = results["ratings"][iteration_idx - 1] if results["ratings"] else "N/A"
                    rating_text = results["rating_texts"][iteration_idx - 1] if results["rating_texts"] else "N/A"
                    transcripts_html += f"""
                    <div class="response-box">&#x1F916; Rating: {rating} (raw response: "{rating_text}")</div>
                    """

        html_content += transcripts_html

    # Close the HTML tags
    html_content += """
    </body>
    </html>
    """

    return html_content

def generate_experiment_xlsx(
    settings_dict,
    chat_data,
    analysis_data,
    chat_results,
    plot_base64
):
    import pandas as pd
    from io import BytesIO
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    import base64

    # Create a new workbook
    wb = Workbook()
    ws_settings = wb.active
    ws_settings.title = 'Settings'

    # Write settings to the first sheet
    row_num = 1
    for key, value in settings_dict.items():
        ws_settings.cell(row=row_num, column=1, value=key)
        ws_settings.cell(row=row_num, column=2, value=value)
        row_num += 1

    # Write chat data to a new sheet
    ws_chat_data = wb.create_sheet(title='Chat Data')
    row_num = 1
    for chat_index, chat in enumerate(chat_data, start=1):
        ws_chat_data.cell(row=row_num, column=1, value=f"Chat {chat_index}")
        row_num += 1
        ws_chat_data.cell(row=row_num, column=1, value='System Message')
        ws_chat_data.cell(row=row_num, column=2, value=chat.get('system_message', ''))
        row_num += 1
        ws_chat_data.cell(row=row_num, column=1, value='Role')
        ws_chat_data.cell(row=row_num, column=2, value='Content')
        row_num += 1
        for msg in chat.get('messages', []):
            ws_chat_data.cell(row=row_num, column=1, value=msg['role'])
            ws_chat_data.cell(row=row_num, column=2, value=msg['content'])
            row_num += 1
        row_num += 1  # Blank row between chats

    # Write analysis data to a new sheet
    ws_analysis = wb.create_sheet(title='Analysis Data')
    for row in analysis_data:
        ws_analysis.append(row)

    # Add plot image if available
    if plot_base64:
        img_data = base64.b64decode(plot_base64)
        from openpyxl.drawing.image import Image as XLImage
        import io

        # Create an in-memory bytes buffer for the image file
        img_file = io.BytesIO(img_data)
        img_file.seek(0)

        # Create an image object
        img = XLImage(img_file)
        img.width = img.width * 0.5  # Adjust size if needed
        img.height = img.height * 0.5

        # Add the image to the worksheet
        ws_analysis.add_image(img, f'A{len(analysis_data)+2}')

    # Write transcripts to a new sheet
    ws_transcripts = wb.create_sheet(title='Transcripts')
    for chat_index in sorted(chat_results.keys()):
        ws_transcripts.append([f"Chat {chat_index}"])
        ws_transcripts.append([])  # Blank line
        for iteration_idx, messages in enumerate(chat_results[chat_index]["messages_per_iteration"], start=1):
            ws_transcripts.append([f"Iteration {iteration_idx}"])
            ws_transcripts.append(['Role', 'Content'])
            for msg in messages:
                ws_transcripts.append([msg['role'], msg['content']])
            ws_transcripts.append(['', ''])  # Blank line between messages
        ws_transcripts.append(['', ''])  # Blank line between chats

    # Save the workbook to a BytesIO object and return bytes
    output = BytesIO()
    wb.save(output)
    xlsx_data = output.getvalue()
    return xlsx_data
