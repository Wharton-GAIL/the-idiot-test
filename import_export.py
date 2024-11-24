import io
import json
import pandas as pd
from xlsxwriter.utility import xl_col_to_name

def load_schema(schema_path='schema.json'):
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        return schema
    except FileNotFoundError:
        raise FileNotFoundError(f"Schema file '{schema_path}' not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from the schema file: {e}")

def validate_value(value, expected_type):
    try:
        if expected_type == 'bool':
            if isinstance(value, str):
                return value.strip().lower() in ('true', '1')
            return bool(value)
        elif expected_type == 'int':
            return int(value)
        elif expected_type == 'float':
            return float(value)
        elif expected_type == 'str':
            return str(value)
        elif expected_type == 'list':
            if isinstance(value, list):
                return value
            raise TypeError(f"Expected list, got {type(value).__name__}")
        else:
            raise TypeError(f"Unsupported type: {expected_type}")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid value '{value}': {e}")

def validate_settings(settings, schema):
    for key, value in settings.items():
        if key in schema:
            expected_type = schema[key]['type']
            if expected_type == 'bool' and not isinstance(value, bool):
                raise TypeError(f"Expected boolean for '{key}', got {type(value).__name__}")
            elif expected_type == 'int' and not isinstance(value, int):
                raise TypeError(f"Expected integer for '{key}', got {type(value).__name__}")
            elif expected_type == 'float' and not isinstance(value, float):
                raise TypeError(f"Expected float for '{key}', got {type(value).__name__}")
            elif expected_type == 'str' and not isinstance(value, str):
                raise TypeError(f"Expected string for '{key}', got {type(value).__name__}")
    return True

def generate_settings_xlsx(settings_dict, chat_data, schema_path='schema.json'):
    # Load the schema
    schema = load_schema(schema_path)['settings']

    # Create a dictionary of settings based on the schema
    settings_values = {key: [settings_dict[key]] for key in schema.keys()}

    # Create a DataFrame using the schema titles
    settings = {
        schema[key]['title']: value for key, value in settings_values.items()
    }
    df_settings = pd.DataFrame(settings).transpose().reset_index()
    df_settings.columns = ['Title', 'Value']

    # Create a BytesIO stream to hold the Excel file in memory
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Define formats
        bold_format = workbook.add_format({'bold': True})
        left_align_format = workbook.add_format({'align': 'left'})
        wrap_format = workbook.add_format({'text_wrap': True, 'align': 'left'})

        # First sheet: Settings
        df_settings.to_excel(writer, sheet_name='Settings', index=False, header=False, startrow=1)
        worksheet_settings = writer.sheets['Settings']

        # Write headers with bold format at the first row (row 0)
        worksheet_settings.write(0, 0, 'Title', bold_format)
        worksheet_settings.write(0, 1, 'Value', bold_format)

        # Set Column A and B widths based on the longest content
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

            # Add system message
            chat_records.append({
                "Chat": f"Chat {idx}",
                "Type": "System Message",
                "Content": system_message
            })

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
                    chat_records.append({
                        "Chat": f"Chat {idx}",
                        "Type": f"Response {prompt_counter}",
                        "Content": response_content
                    })

        df_chat = pd.DataFrame(chat_records)

        # Define the desired order for 'Type'
        initial_type_order = ["System Message", "Evaluation Rubric"]
        prompt_types = df_chat['Type'].str.extract(r'Prompt (\d+)').dropna()[0].astype(int).unique()
        prompt_types.sort()
        for num in prompt_types:
            initial_type_order.append(f"Prompt {num}")
            initial_type_order.append(f"Response {num}")

        # Ensure the DataFrame respects this exact order
        df_chat['Type'] = pd.Categorical(df_chat['Type'], categories=initial_type_order, ordered=True)

        # Sort the DataFrame by Chat and then by the Type order
        df_chat = df_chat.sort_values(['Chat', 'Type'])

        # Pivot the DataFrame to have 'Type' as rows and each chat as a separate column
        pivot_df = df_chat.pivot(index='Type', columns='Chat', values='Content')

        # Reset index to have 'Type' as a column
        pivot_df.reset_index(inplace=True)
        pivot_df.to_excel(writer, sheet_name='Prompts', index=False, header=True)
        worksheet_chat = writer.sheets['Prompts']

        # Adjust column widths and formats
        # First column: Auto-resize based on the longest text
        max_length_type = df_chat['Type'].astype(str).map(len).max()
        optimal_width_type = max_length_type + 2
        worksheet_chat.set_column('A:A', optimal_width_type, wrap_format)

        # Subsequent columns: Set width to 75 and enable word wrap
        for idx, chat in enumerate(sorted(df_chat['Chat'].unique()), start=2):
        # Subsequent columns: Set width to 75 and enable word wrap
            col_letter = xl_col_to_name(idx - 1)
            worksheet_chat.set_column(f'{col_letter}:{col_letter}', 75, wrap_format)

    output.seek(0)
    return output

def import_settings_xlsx(xlsx_file, schema_path='schema.json'):
    try:
        # Load the schema
        schema = load_schema(schema_path)
        settings_schema = schema['settings']
        chat_schema = schema['chat_data']

        # Create a reverse mapping from titles to keys
        title_to_key = {v['title']: k for k, v in settings_schema.items()}

        # Read the 'Settings' sheet
        df_settings = pd.read_excel(
            xlsx_file,
            sheet_name='Settings',
            header=None,
            names=['Title', 'Value'],
            skiprows=1
        )

        settings = {}
        for _, row in df_settings.iterrows():
            title = row['Title']
            value = row['Value']
            if title in title_to_key:
                key = title_to_key[title]
                expected_type = settings_schema[key]['type']
                try:
                    # Convert the value to the expected type
                    converted_value = validate_value(value, expected_type)
                    settings[key] = converted_value
                except ValueError as e:
                    raise ValueError(f"Invalid value for '{title}': {e}")
            else:
                # Ignore unrecognized settings
                continue

        # Validate settings
        validate_settings(settings, settings_schema)

        # Read the 'Prompts' sheet (previously 'Chat Data')
        df_chat = pd.read_excel(xlsx_file, sheet_name='Prompts')

        # Set 'Type' as the index
        df_chat.set_index('Type', inplace=True)

        # Extract chat columns (e.g., 'Chat 1', 'Chat 2', ...)
        chat_columns = [col for col in df_chat.columns if col.startswith('Chat')]

        chat_data = []
        for chat_col in chat_columns:
            chat = {}
            messages = []

            chat_content = df_chat[chat_col].fillna('')

            # Extract system message
            system_message = chat_content.get('System Message', '').strip()
            chat['system_message'] = system_message

            # Extract evaluation rubric
            rating_prompt_template = chat_content.get('Evaluation Rubric', '').strip()
            chat['rating_prompt_template'] = rating_prompt_template

            # Find all prompt and response types
            prompt_types = [t for t in df_chat.index if t.startswith('Prompt ')]
            response_types = [t for t in df_chat.index if t.startswith('Response ')]

            # Extract prompt numbers
            prompt_numbers = sorted(set(int(t.split(' ')[1]) for t in prompt_types))

            for num in prompt_numbers:
                prompt_key = f'Prompt {num}'
                response_key = f'Response {num}'

                prompt_content = chat_content.get(prompt_key, '').strip()
                response_content = chat_content.get(response_key, '').strip()

                # Handle '[AI Responds]' as a blank entry
                if response_content == '[AI Responds]':
                    response_content = ''

                # Add prompt to messages if not blank
                if prompt_content:
                    messages.append({"role": "user", "content": prompt_content})

                # Add response to messages if not blank
                if response_content:
                    messages.append({"role": "assistant", "content": response_content})

            chat['messages'] = messages
            chat_data.append(chat)

        # Validate chat_data
        validate_chat_data(chat_data, chat_schema)

        settings['chat_data'] = chat_data
        return settings
    except FileNotFoundError:
        raise FileNotFoundError("The provided XLSX file does not contain the required sheets.")
    except pd.errors.EmptyDataError:
        raise ValueError("The XLSX file is empty or corrupted.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during import: {e}")

def validate_chat_data(chat_data, schema):
    for chat in chat_data:
        # Validate system_message
        if not isinstance(chat.get("system_message", ""), str):
            raise TypeError("system_message must be a string.")
        
        # Validate rating_prompt_template
        if chat.get("rating_prompt_template") is not None and not isinstance(chat["rating_prompt_template"], str):
            raise TypeError("rating_prompt_template must be a string or None.")
        
        # Validate messages
        messages = chat.get("messages", [])
        if not isinstance(messages, list):
            raise TypeError("messages must be a list.")
        
        for message in messages:
            if not isinstance(message.get("role"), str):
                raise TypeError("Each message's role must be a string.")
            if not isinstance(message.get("content"), str):
                raise TypeError("Each message's content must be a string.")
    return True