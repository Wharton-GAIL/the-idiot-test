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
                if value.strip().lower() in ('true', '1'):
                    return True
                elif value.strip().lower() in ('false', '0'):
                    return False
                else:
                    raise ValueError(f"Cannot convert string '{value}' to boolean.")
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
    # Set '10x iterations' to False if it's not present
    if '10x_iterations' not in settings:
        settings['10x_iterations'] = False

    for key, key_schema in schema.items():
        title = schema[key]['title'] if key in schema else key
        if key not in settings:
            raise ValueError(f"Missing required setting '{title}'.")
        value = settings[key]
        expected_type = key_schema['type']
        actual_type = type(value).__name__
        if expected_type == 'bool' and not isinstance(value, bool):
            raise TypeError(f"Expected boolean for '{title}', got {actual_type}.")
        elif expected_type == 'int' and not isinstance(value, int):
            raise TypeError(f"Expected integer for '{title}', got {actual_type}.")
        elif expected_type == 'float' and not isinstance(value, float):
            raise TypeError(f"Expected float for '{title}', got {actual_type}.")
        elif expected_type == 'str' and not isinstance(value, str):
            raise TypeError(f"Expected string for '{title}', got {actual_type}.")
        elif expected_type == 'list' and not isinstance(value, list):
            raise TypeError(f"Expected list for '{title}', got {actual_type}.")
    return True


def generate_settings_xlsx(settings_dict, chat_data, schema_path='schema.json', validate_chat=True):
    # Validate settings_dict and chat_data before exporting
    schema = load_schema(schema_path)
    settings_schema = schema['settings']
    chat_schema = schema['chat_data']
    validate_settings(settings_dict, settings_schema)
    if validate_chat:
        validate_chat_data(chat_data, schema)

    # Create a dictionary of settings based on the schema
    settings_values = {key: [settings_dict[key]] for key in settings_schema.keys()}

    # Create a DataFrame using the schema titles
    settings = {
        settings_schema[key]['title']: value for key, value in settings_values.items()
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
                "chat": f"Chat {idx}",
                "type": "System Message",
                "content": system_message,
                "row": None
            })

            # Add evaluation rubric
            chat_records.append({
                "chat": f"Chat {idx}",
                "type": "Evaluation Rubric",
                "content": rating_prompt,
                "row": None
            })

            # Add chat (user prompts and assistant responses)
            messages = chat.get("messages", [])
            prompt_counter = 0
            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "").strip()
                
                if role == "user":
                    prompt_counter += 1
                    number = prompt_counter
                    message_type = f"Prompt {number}"
                elif role == "assistant":
                    number = prompt_counter
                    message_type = f"Response {number}"
                else:
                    raise ValueError(f"Unexpected role: {role}")

                chat_records.append({
                    "chat": f"Chat {idx}",
                    "type": message_type,
                    "content": content if content else "[AI Responds]",
                    "row": None,
                    "role": role,
                    "number": number
                })

        df_chat = pd.DataFrame(chat_records)

        # Define the desired order for 'type'
        initial_type_order = ["System Message", "Evaluation Rubric"]
        prompt_numbers = df_chat['type'].str.extract(r'Prompt (\d+)').dropna()[0].astype(int).unique()
        prompt_numbers.sort()
        for num in prompt_numbers:
            initial_type_order.append(f"Prompt {num}")
            initial_type_order.append(f"Response {num}")

        # Ensure the DataFrame respects this exact order
        df_chat['type'] = pd.Categorical(df_chat['type'], categories=initial_type_order, ordered=True)

        # Sort the DataFrame by 'chat' and then by the 'type' order
        df_chat = df_chat.sort_values(['chat', 'type'])

        # Check for duplicates before pivoting
        duplicates = df_chat[df_chat.duplicated(subset=['type', 'chat'], keep=False)]
        if not duplicates.empty:
            duplicate_details = duplicates.to_string()
            raise ValueError(f"Found duplicate type/chat combinations, which is not allowed:\n{duplicate_details}")

        # Pivot the DataFrame to have 'type' as rows and each chat as a separate column
        pivot_df = df_chat.pivot(index='type', columns='chat', values='content')

        # Reset index to have 'type' as a column
        pivot_df.reset_index(inplace=True)
        pivot_df.to_excel(writer, sheet_name='Prompts', index=False, header=True)
        worksheet_chat = writer.sheets['Prompts']

        # Adjust column widths and formats
        # First column: Auto-resize based on the longest text
        max_length_type = df_chat['type'].astype(str).map(len).max()
        optimal_width_type = max_length_type + 2
        worksheet_chat.set_column('A:A', optimal_width_type, wrap_format)

        # Subsequent columns: Set width to 75 and enable word wrap
        for idx, chat in enumerate(sorted(df_chat['chat'].unique()), start=2):
            # Subsequent columns: Set width to 75 and enable word wrap
            col_letter = xl_col_to_name(idx - 1)
            worksheet_chat.set_column(f'{col_letter}:{col_letter}', 75, wrap_format)

        # Third sheet: Friendly Prompt
        friendly_records = []
        for idx, chat in enumerate(chat_data, start=1):
            # Add system message
            system_message = chat.get("system_message", "").strip()
            if system_message:
                friendly_records.append({
                    "chat": f"Chat {idx}",
                    "content": f"⚙️ {system_message}"
                })

            # Add chat messages
            messages = chat.get("messages", [])
            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "").strip()
                
                if role == "user":
                    emoji = "🧑"
                elif role == "assistant":
                    emoji = "🤖"
                    if not content:
                        content = "[AI Responds]"
                else:
                    continue

                friendly_records.append({
                    "chat": f"Chat {idx}",
                    "content": f"{emoji} {content}"
                })

        df_friendly = pd.DataFrame(friendly_records)
        
        # Pivot the DataFrame to have each chat as a separate column
        pivot_friendly = df_friendly.pivot_table(
            index=df_friendly.groupby('chat').cumcount(),
            columns='chat',
            values='content',
            aggfunc='first'
        )

        # Write to Excel
        pivot_friendly.to_excel(writer, sheet_name='Friendly Prompt', index=False)
        worksheet_friendly = writer.sheets['Friendly Prompt']

        # Format the columns
        for idx, chat in enumerate(sorted(df_friendly['chat'].unique())):
            col_letter = xl_col_to_name(idx)
            worksheet_friendly.set_column(f'{col_letter}:{col_letter}', 75, wrap_format)

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
        for idx, row in df_settings.iterrows():
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
                    raise ValueError(f"Row {idx + 2} - Invalid value for '{title}': {e}")
            else:
                raise ValueError(f"Row {idx + 2} - Unrecognized setting '{title}'.")

        # Validate settings
        validate_settings(settings, settings_schema)

        # Read the 'Prompts' sheet
        df_chat = pd.read_excel(xlsx_file, sheet_name='Prompts')
        
        # Normalize column names to handle case-insensitive 'type'
        df_chat.columns = [col.lower() if col.lower() == 'type' else col for col in df_chat.columns]
        if 'type' not in df_chat.columns:
            raise ValueError("Could not find 'type' or 'Type' column in Prompts sheet")

        # Reset index to get row numbers (DataFrame index corresponds to Excel rows minus header)
        df_chat.reset_index(inplace=True)
        df_chat['row_number'] = df_chat.index + 2  # Adjusting for 0-based & header row

        # Set 'type' as the index
        df_chat.set_index('type', inplace=True)

        # Extract chat columns (e.g., 'Chat 1', 'Chat 2', ...)
        chat_columns = [col for col in df_chat.columns if col.startswith('Chat')]

        chat_data = []
        for chat_idx, chat_col in enumerate(chat_columns, start=1):
            chat = {}
            chat_content = df_chat[chat_col].fillna('')
            row_numbers = df_chat['row_number']

            # Extract system message
            system_message = chat_content.get('System Message', '').strip()
            chat['system_message'] = system_message

            # Extract evaluation rubric
            rating_prompt_template = chat_content.get('Evaluation Rubric', '').strip()
            chat['rating_prompt_template'] = rating_prompt_template

            # Find all prompt and response types
            message_types = [t for t in df_chat.index if t.startswith('Prompt') or t.startswith('Response')]

            # Collect all messages
            message_entries = []
            for message_type in message_types:
                if message_type.startswith('Prompt'):
                    role = 'user'
                elif message_type.startswith('Response'):
                    role = 'assistant'
                else:
                    raise ValueError(f"Unexpected message type: {message_type} in chat {chat_idx}")

                content = chat_content.get(message_type, '').strip()
                number = int(message_type.split(' ')[1])
                row = row_numbers.get(message_type)

                if role == 'assistant' and content == '[AI Responds]':
                    content = ''

                message_entries.append({
                    "role": role,
                    "content": content,
                    "number": number,
                    "row": row,
                    "type": message_type,
                    "chat": chat_col
                })

            # Trim trailing blank messages
            last_non_blank_index = None
            for idx, message in reversed(list(enumerate(message_entries))):
                if message['content']:
                    last_non_blank_index = idx
                    break

            if last_non_blank_index is not None:
                messages = message_entries[:last_non_blank_index+1]
            else:
                messages = []

            chat['messages'] = messages
            chat_data.append(chat)

        # Validate chat_data
        validate_chat_data(chat_data, chat_schema)

        settings['chat_data'] = chat_data
        return settings

    except FileNotFoundError as e:
        raise FileNotFoundError(f"The provided XLSX file does not contain the required sheets: {e}")
    except pd.errors.EmptyDataError:
        raise ValueError("The XLSX file is empty or corrupted.")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Validation error: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during import: {e}")

def validate_chat_data(chat_data, schema):
    for chat_idx, chat in enumerate(chat_data, start=1):
        # Validate system_message
        if not isinstance(chat.get("system_message", ""), str):
            raise TypeError(f"Chat {chat_idx}: 'system_message' must be a string.")

        # Validate rating_prompt_template
        if chat.get("rating_prompt_template") is not None and not isinstance(chat["rating_prompt_template"], str):
            raise TypeError(f"Chat {chat_idx}: 'rating_prompt_template' must be a string or None.")

        # Validate messages
        messages = chat.get("messages", [])
        if not isinstance(messages, list):
            raise TypeError(f"Chat {chat_idx}: 'messages' must be a list.")

        # Expected sequence: Prompt 1, Response 1, Prompt 2, Response 2, etc.
        expected_number = 1
        i = 0
        while i < len(messages):
            # Expect a user prompt
            message = messages[i]
            if message.get("role") != "user":
                raise ValueError(
                    f"Row {message.get('row')}, Type '{message.get('type')}', Chat '{message.get('chat')}': Expected a 'prompt' (user), got '{message.get('role')}'."
                )
            if not isinstance(message.get("content"), str):
                raise TypeError(
                    f"Row {message.get('row')}, Type '{message.get('type')}', Chat '{message.get('chat')}': 'content' must be a string."
                )

            if message.get("number") != expected_number:
                raise ValueError(
                    f"Row {message.get('row')}, Type '{message.get('type')}', Chat '{message.get('chat')}': Expected prompt number {expected_number}, got {message.get('number')}."
                )

            i += 1  # Move to the next message

            # Check if there is an assistant response
            if i < len(messages):
                message = messages[i]
                if message.get("role") != "assistant":
                    raise ValueError(
                        f"Row {message.get('row')}, Type '{message.get('type')}', Chat '{message.get('chat')}': Expected a 'response' (assistant), got '{message.get('role')}'."
                    )
                if not isinstance(message.get("content"), str):
                    raise TypeError(
                        f"Row {message.get('row')}, Type '{message.get('type')}', Chat '{message.get('chat')}': 'content' must be a string."
                    )

                if message.get("number") != expected_number:
                    raise ValueError(
                        f"Row {message.get('row')}, Type '{message.get('type')}', Chat '{message.get('chat')}': Expected response number {expected_number}, got {message.get('number')}."
                    )

                i += 1  # Move to the next message

            expected_number += 1

        # Ensure that the last message is not an assistant response
        if len(messages) >= 1 and messages[-1].get("role") == "assistant":
            message = messages[-1]
            print(f"Full chat: \n{chat}")
            raise ValueError(
                f"Row {message.get('row')}, Type '{message.get('type')}', Chat '{message.get('chat')}': Conversation cannot end with a response (assistant)."
            )

    return True