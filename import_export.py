import io
import pandas as pd
from xlsxwriter.utility import xl_col_to_name

def generate_settings_xlsx(
    number_of_iterations,
    model_response,
    temperature_response,
    model_rating,
    temperature_rating,
    analyze_rating,
    analyze_length,
    show_transcripts,
    chat_data
):
    """
    Generate an XLSX file containing settings and comprehensive chat data with titles in the first column.

    Args:
        number_of_iterations (int): Number of iterations.
        model_response (str): Model used for response generation.
        temperature_response (float): Temperature for response generation.
        model_rating (str): Model used for rating.
        temperature_rating (float): Temperature for rating.
        analyze_rating (bool): Whether to use AI to analyze ratings.
        analyze_length (bool): Whether to analyze the length of responses.
        show_transcripts (bool): Whether to add a table of all responses.
        chat_data (list): List of chat data dictionaries.

    Returns:
        BytesIO: In-memory binary stream of the XLSX file.
    """
    # Create a BytesIO stream to hold the Excel file in memory
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Define formats
        bold_format = workbook.add_format({'bold': True})
        left_align_format = workbook.add_format({'align': 'left'})
        wrap_format = workbook.add_format({'text_wrap': True, 'align': 'left'})
        
        # First sheet: Settings
        settings = {
            "Number of Iterations": [number_of_iterations],
            "Model for Response Generation": [model_response],
            "Temperature for Response Generation": [temperature_response],
            "Model for Rating": [model_rating],
            "Use AI to analyze ratings": [analyze_rating],
            "Analyze length of response": [analyze_length],
            "Add table of all responses": [show_transcripts],
            "Temperature for Rating": [temperature_rating]
        }
        print(settings)
        df_settings = pd.DataFrame(settings).transpose().reset_index()
        df_settings.columns = ['Title', 'Value']
        print(df_settings)
        
        df_settings.to_excel(writer, sheet_name='Settings', index=False, header=False, startrow=1)
        
        worksheet_settings = writer.sheets['Settings']
        
        # Write headers with bold format at the first row (row 0)
        worksheet_settings.write(0, 0, 'Title', bold_format)
        worksheet_settings.write(0, 1, 'Value', bold_format)
        
        # Set Column A width based on the longest title
        max_length_title = df_settings['Title'].astype(str).map(len).max()
        max_length_value = df_settings['Value'].astype(str).map(len).max()
        
        # Calculate optimal widths with some padding
        optimal_width_title = max_length_title + 2
        optimal_width_value = max_length_value + 2
        
        # Set column widths to auto-adjust based on content
        worksheet_settings.set_column('A:A', optimal_width_title)
        worksheet_settings.set_column('B:B', optimal_width_value, left_align_format)

        # Second sheet: Chat Data
        chat_records = []
        for idx, chat in enumerate(chat_data, start=1):
            system_message = chat.get("system_message", "")
            rating_prompt = chat.get("rating_prompt_template", "")
            
            # Add system prompt
            chat_records.append({
                "Chat": f"Chat {idx}",
                "Type": "System Prompt",
                "Content": system_message
            })
            
            # Add evaluation rubric if available
            if rating_prompt:
                chat_records.append({
                    "Chat": f"Chat {idx}",
                    "Type": "Evaluation Rubric",
                    "Content": rating_prompt
                })
            
            # Add user prompts and assistant responses
            messages = chat.get("messages", [])
            prompt_counter = 0
            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "").strip()
                if role == "user":
                    prompt_counter += 1
                    # Add prompt
                    chat_records.append({
                        "Chat": f"Chat {idx}",
                        "Type": f"Prompt {prompt_counter}",
                        "Content": content
                    })
                elif role == "assistant":
                    # Replace blank responses with "[AI Responds]" to disambiguate
                    response_content = content if content else "[AI Responds]"
                    # Add response
                    chat_records.append({
                        "Chat": f"Chat {idx}",
                        "Type": f"Response {prompt_counter}",
                        "Content": response_content
                    })
        
        df_chat = pd.DataFrame(chat_records)
        
        # Define the desired order for 'Type'
        type_order = ["System Prompt", "Evaluation Rubric"]
        
        # Determine the maximum number of prompt-response pairs
        max_prompt_response = df_chat['Type'].str.extract(r'Prompt (\d+)').dropna()[0].astype(int).max()
        
        # Add prompt-response pairs in the correct order
        for i in range(1, int(max_prompt_response) + 1):
            type_order.append(f"Prompt {i}")
            type_order.append(f"Response {i}")
        
        # Ensure the DataFrame respects this exact order
        df_chat['Type'] = pd.Categorical(df_chat['Type'], categories=type_order, ordered=True)
        
        # Sort the DataFrame by Chat and then by the Type order
        df_chat = df_chat.sort_values(['Chat', 'Type'])
        
        # Pivot the DataFrame to have 'Type' as rows and each chat as a separate column
        pivot_df = df_chat.pivot(index='Type', columns='Chat', values='Content')
        
        # Reset index to have 'Type' as a column
        pivot_df.reset_index(inplace=True)
        
        # Write the pivoted DataFrame to Excel
        pivot_df.to_excel(writer, sheet_name='Chat Data', index=False, header=True)
        
        worksheet_chat = writer.sheets['Chat Data']
        
        # Adjust column widths and formats
        # First column: Auto-resize based on the longest text
        max_length_type = df_chat['Type'].astype(str).map(len).max()
        optimal_width_type = max_length_type + 2  # Adding padding
        worksheet_chat.set_column('A:A', optimal_width_type, wrap_format)
        
        # Subsequent columns: Set width to 75 and enable word wrap
        for idx, chat in enumerate(sorted(df_chat['Chat'].unique()), start=2):
            col_letter = xl_col_to_name(idx - 1)  # Adjusted index for zero-based columns
            worksheet_chat.set_column(f'{col_letter}:{col_letter}', 75, wrap_format)

    # Seek to the beginning of the stream
    output.seek(0)
    return output