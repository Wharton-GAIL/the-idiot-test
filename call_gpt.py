"""
# call_gpt documentation

call_gpt sends a query to an LLM and returns a response. Syntax:
def call_gpt(query: Union[str, List[Dict[str, str]]], settings: Optional[Dict[str, Any]] = None, return_pricing: bool = False, return_dict: bool = False, functions: Optional[List[dict]] = None, function_call: Optional[dict] = None, logit_bias: Optional[dict] = None, stop_sequences: Optional[List[str]] = None, system_prompt: Optional[str] = None, openai_api_key: Optional[str] = None, anthropic_api_key: Optional[str] = None, google_api_key: Optional[str] = None) -> Union[str, Dict[str, Any], Tuple[str, float], Tuple[Dict[str, Any], float]]:

**Parameters:**
- `query`: A string will be converted to a single user message. 
  If it is a list, format it like this:
    [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Name a color."},
        {"role": "assistant", "content": "Red"},
        {"role": "user", "content": "Another?"},
    ]
- `settings`: If omitted, sensible defaults will be used. If included, it overrides the defaults for some or all of these keys:
  - `"model"`: Can be `gpt-3.5-turbo`, `gpt-4-turbo`, or others.
  - `"temperature"`: Default is `0.7`.
  - `"n"`: Number of replies.
  - `"max_tokens"`: Default is the maximum allowable.
  - `"min_reply_tokens"`: Default is `max_tokens / 4`.
- `return_pricing`: If `True`, returns the cost of the call as a float representing USD.
- `return_dict`: If `True`, returns the full API response as a dictionary; otherwise, just the reply string.
- `functions`: List of dictionaries specifying function(s) the LLM can call.
  - `name`: Name of the function.
  - `parameters`: Object detailing the function's arguments and their types.
- `function_call`: Dictionary to invoke a function from the API call.
  - `name`: Name of the function to call.
  - `args`: Arguments for the function.
- `logit_bias`: Token IDs from tiktoken are keys. Biases from `-100` to `100` are valid.
- `stop_sequences`: List of strings that will halt LLM output if generated, without generating them.
- `system_prompt`: Optional system prompt to set the behavior of the model.
- `openai_api_key`: Optional OpenAI API key. If not provided, the environment variable `OPENAI_API_KEY` is used.
- `anthropic_api_key`: Optional Anthropic API key. If not provided, the environment variable `ANTHROPIC_API_KEY` is used.
- `google_api_key`: Optional Google API key. If not provided, the environment variable `GOOGLE_API_KEY` is used.
    
**Returns:**
- By default, returns reply content as a string.
- If `return_dict` is `True`, returns the full API response as a dictionary.
- If `return_pricing` is `True`, returns a tuple with the first element being the reply (string or dictionary) and the second being the cost.
"""
import os
import sys
import openai
import tiktoken
import google.generativeai as genai  # pip install google-generativeai
from google.generativeai.types.content_types import to_contents
import google.api_core.exceptions
import google.auth.exceptions
import google.generativeai.types
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import time
import inspect
import unittest
import requests
import random
from func_timeout import func_timeout, FunctionTimedOut
from requests.exceptions import ReadTimeout, ConnectTimeout
from typing import *
from dotenv import load_dotenv
import json
import traceback

if os.name == 'nt':  # Windows
    sys.path.append('..\\dan-tools')
else:  # Linux
    sys.path.append('../dan-tools')
from log_love import setup_logging

logger = None
logger = setup_logging(logger)

load_dotenv()

# Define model details. Pricing is per 1M tokens. Rate limits are request/min, but not yet implemented.
MODELS = {
    "o1-preview":          {"pricing": (15, 60), "tokens": (128000, 4096), "rate_limit": 10000},
    "o1-mini":             {"pricing": (3, 12), "tokens": (128000, 4096), "rate_limit": 10000},
    "gpt-4":               {"pricing": (5, 15), "tokens": (8192, 8192), "rate_limit": 10000},
    "gpt-4o":              {"pricing": (10, 30), "tokens": (128000, 4096), "rate_limit": 10000},
    "gpt-4o-mini":         {"pricing": (0.15, 0.60), "tokens": (128000, 16384), "rate_limit": 10000},
    "gpt-4-turbo":         {"pricing": (10, 30), "tokens": (128000, 4096), "rate_limit": 10000},
    "gpt-3.5-turbo":       {"pricing": (.5, 1.5), "tokens": (4096, 4096), "rate_limit": 10000},
    "gpt-3.5-turbo-0613":  {"pricing": (1.5, 2), "tokens": (4096, 4096), "rate_limit": 10000},
    "gpt-3.5-turbo-16k":   {"pricing": (3, 4), "tokens": (16000, 16000), "rate_limit": 10000},
    "claude-instant-v1":   {"pricing": (1.63, 5.51), "tokens": (9000, 9000), "rate_limit": 60},
    "claude-v1":           {"pricing": (11.02, 32.68), "tokens": (9000, 9000), "rate_limit": 60},
    "claude-3-opus-20240229":   {"pricing": (15, 75), "tokens": (200000, 4000), "rate_limit": 50},
    "claude-3-sonnet-20240229": {"pricing": (3, 15), "tokens": (200000, 4000), "rate_limit": 50},
    "claude-3-5-sonnet-20240620": {"pricing": (3, 15), "tokens": (200000, 4000), "rate_limit": 50}, 
    "claude-3-5-sonnet-20241022": {"pricing": (3, 15), "tokens": (200000, 4000), "rate_limit": 50},
    "claude-3-haiku-20240307":  {"pricing": (0.25, 1.25), "tokens": (200000, 4000), "rate_limit": 50},
    "gemini-1.5-flash":    {"pricing": (0.075, 0.30), "tokens": (1048576, 8192), "rate_limit": 1000},
    "gemini-1.5-pro":      {"pricing": (3.50, 10.50), "tokens": (2097152, 8192), "rate_limit": 360}
}

def expand_model_shortcut(model: str) -> str:
    if model == "gpt-4":
        return "gpt-4o"
    if model == "claude":
        return "claude-3-5-sonnet-20241022"
    if model == "opus":
        return "claude-3-opus-20240229"
    if model == "sonnet":
        return "claude-3-5-sonnet-20241022"
    if model == "haiku":
        return "claude-3-haiku-20240307"
    if model == "gpt-4-turbo-preview":
        return "gpt-4-turbo"
    if model == "gemini":
        return "gemini-1.5-flash"
    return model

MISSING = object()

# Given exactly one parameter which is an API key (possibly value None), returns the API key. It's the key provided if present, or the environment variable if not.
def get_api_key(
    openai_api_key: Optional[str] = MISSING,
    anthropic_api_key: Optional[str] = MISSING,
    google_api_key: Optional[str] = MISSING
) -> str:
    # Count how many parameters were explicitly passed (even if None)
    passed_params = sum(
        param is not MISSING for param in [openai_api_key, anthropic_api_key, google_api_key]
    )
    
    if passed_params == 0:
        raise ValueError("At least one API key parameter must be specified")    
    if passed_params > 1:
        raise ValueError("Only one API key should be specified at a time")

    # Handle each possible key
    if openai_api_key is not MISSING:
        if openai_api_key:
            return openai_api_key
        env_key = os.getenv('OPENAI_API_KEY')
        if env_key:
            return env_key
        raise ValueError("OPENAI_API_KEY is required but not provided and not set in environment variables")
    
    if anthropic_api_key is not MISSING:
        if anthropic_api_key:
            return anthropic_api_key
        env_key = os.getenv('ANTHROPIC_API_KEY')
        if env_key:
            return env_key
        raise ValueError("ANTHROPIC_API_KEY is required but not provided and not set in environment variables")
    
    if google_api_key is not MISSING:
        if google_api_key:
            return google_api_key
        env_key = os.getenv('GOOGLE_API_KEY')
        if env_key:
            return env_key
        raise ValueError("GOOGLE_API_KEY is required but not provided and not set in environment variables")

    raise ValueError("No API key found")  # Should never reach here

def get_model_input_token_max(model: str) -> int:
    try: 
        return MODELS[model]["tokens"][0]
    except KeyError:
        logger.error(f"Invalid model in get_model_input_token_max: {model}")
        raise ValueError(f"Invalid model in get_model_input_token_max: {model}")

def get_model_output_token_max(model: str) -> int:
    try: 
        return MODELS[model]["tokens"][1]
    except KeyError:
        logger.error(f"Invalid model: in get_model_output_token_max: {model}")
        raise ValueError(f"Invalid model in get_model_output_token_max: {model}")
    
def get_model_token_max(model: str) -> int:
    logger.error("This function is deprecated. Update to call get_model_input_token_max or get_model_output_token_max")
    return get_model_output_token_max(model)
    
def get_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    input_price_per_token, output_price_per_token = (price / 1000000 for price in MODELS[model]["pricing"])
    input_cost = input_price_per_token * input_tokens
    output_cost = output_price_per_token * output_tokens
    total_cost = input_cost + output_cost
    return total_cost

def get_tokens(to_tokenize: str, model: str, anthropic_api_key: Optional[str] = None, google_api_key: Optional[str] = None) -> Optional[int]:
    model = expand_model_shortcut(model)
    if not isinstance(to_tokenize, str):
        raise ValueError("Input to tokenize must be a string")
    if model.startswith('gemini-'):
        genai.configure(api_key=google_api_key)
        genai_model = genai.GenerativeModel(model)
        return genai_model.count_tokens(to_tokenize).total_tokens
    elif model.startswith('claude-'):
        anthropic_api_key = get_api_key(anthropic_api_key=anthropic_api_key)
        url = "https://api.anthropic.com/v1/messages/count_tokens"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "token-counting-2024-11-01"
        }
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": to_tokenize}
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result.get('input_tokens')
    elif model.startswith('gpt-') or model.startswith('o1-'):
        try:
            # Map 'o1-' models to 'gpt-4' for tokenization
            if model.startswith('o1-'):
                encoding = tiktoken.encoding_for_model('gpt-4')
            else:
                encoding = tiktoken.encoding_for_model(model)
        except Exception as e:
            raise ValueError(f"Tiktoken Error: {e}")
        try:
            num_tokens = len(encoding.encode(to_tokenize))
        except Exception as e:
            logger.error(f"Tiktoken encoding error: {e}")
            return None
        return num_tokens
    else:
        raise ValueError(f"Invalid model: {model}")

# very helpful: https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/chatgpt?pivots=programming-language-chat-completions
def num_tokens_from_messages(
    messages: List[Dict[str, str]],
    model: str,
    functions: Optional[List[dict]] = None,
    function_call: Optional[dict] = None,
    system_prompt: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None
) -> int:
    if model.startswith('gemini-'):
        genai.configure(api_key=google_api_key)
        genai_model = genai.GenerativeModel(model)
        # Convert messages to Gemini format
        gemini_content = []
        for message in messages:
            if message['role'] == 'user':
                gemini_content.append({"role": "user", "parts": [{"text": message['content']}]})
            elif message['role'] == 'assistant':
                gemini_content.append({"role": "model", "parts": [{"text": message['content']}]})
            else:
                # Handle system messages or any other roles as user messages
                gemini_content.append({"role": "user", "parts": [{"text": message['content']}]})
        
        return genai_model.count_tokens(gemini_content).total_tokens
    
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # for the role and content keys
        for key, value in message.items():
            try:
                tokens = get_tokens(value, model, anthropic_api_key=anthropic_api_key, google_api_key=google_api_key)
                if tokens is None:
                    logger.error(f"Tokenization failed for the value: {value}")
                    continue
                num_tokens += tokens
            except Exception as e:
                logger.error(f"Error: {e}")
                raise e
            if key == "name":
                num_tokens += -1
    num_tokens += 2  # for the list brackets

    if functions:
        for function in functions:
            num_tokens += get_tokens(str(function), model, anthropic_api_key=anthropic_api_key, google_api_key=google_api_key)  # for the function name

    if function_call:
        num_tokens += get_tokens(str(function_call), model, anthropic_api_key=anthropic_api_key, google_api_key=google_api_key)  # for the function call

    if system_prompt:
        num_tokens += get_tokens(system_prompt, model, anthropic_api_key=anthropic_api_key, google_api_key=google_api_key)
    
    return num_tokens

# Trim messages so that there are at least min_reply_tokens available for responses (based on the model being used).
def trim_messages(messages: List[Dict[str, str]], model: str, min_reply_tokens: Optional[int] = None, functions: Optional[List[dict]] = None, function_call: Optional[dict] = None, system_prompt: Optional[str] = None, anthropic_api_key: Optional[str] = None) -> List[Dict[str, str]]:
    # Reserve for output 1/4 of input tokens or all of output tokens, whichever is smaller
    if min_reply_tokens is None:
        min_reply_tokens = min(get_model_input_token_max(model) / 4, get_model_output_token_max(model))
    messages_tokens = num_tokens_from_messages(messages, model, functions, function_call, system_prompt, anthropic_api_key)
    start_size = messages_tokens
    messages_deleted = 0
    start = 0
    if messages[start]["role"] == "system":
        start = 1
    trimmed_messages = messages.copy()  # Create a copy of messages to avoid modifying the original
    while (
        messages_tokens + min_reply_tokens >= get_model_input_token_max(model)
        and len(trimmed_messages) > start + 1
    ):
        del trimmed_messages[start]
        messages_deleted += 1
        messages_tokens = num_tokens_from_messages(trimmed_messages, model, functions, function_call, system_prompt, anthropic_api_key)
    if messages_deleted > 0:
        logger.info(
            f"TRIMMED: {messages_deleted} messages deleted. Required {min_reply_tokens} tokens for reply in {model} ({get_model_output_token_max(model)} output token max), so messages trimmed from {start_size} to {messages_tokens} tokens. Output can have {get_model_output_token_max(model)-messages_tokens} tokens now."
        )
    else:
        logger.debug(
            f"No messages deleted. Required {min_reply_tokens} tokens for reply in {model} ({get_model_output_token_max(model)} output token max), so messages trimmed from {start_size} to {messages_tokens} tokens. Output can have {get_model_output_token_max(model)-messages_tokens} tokens now."
        )
    return trimmed_messages

def send_llm_request(
    model: str,
    temperature: float,
    messages: List[dict],
    max_tokens: int,
    n: int,
    timeout: float,
    stop_sequences: Optional[List[str]] = None,
    logit_bias: Optional[dict] = None,
    functions: Optional[List[dict]] = None,
    function_call: Optional[dict] = None,
    system_prompt: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None
) -> dict:
    """
    Sends a request to the LLM API.
    
    Note: Special handling for 'o1' and 'o1-mini' models:
    - Uses 'max_completion_tokens' instead of 'max_tokens'
    - Omits temperature, n, and other unsupported parameters
    - Does not support functions, function_call, logit_bias, stop_sequences
    
    This implementation may need to be updated as 'o1' models support more features.
    """
    if model.startswith('gemini-'):
        # Convert messages to Gemini format
        gemini_content = []
        for message in messages:
            if message['role'] == 'user':
                gemini_content.append({"role": "user", "parts": [{"text": message['content']}]})
            elif message['role'] == 'assistant':
                gemini_content.append({"role": "model", "parts": [{"text": message['content']}]})
            else:
                logger.error(f"Unexpected message role: {message['role']}. Appending it as a user message.")
                gemini_content.append({"role": "user", "parts": [{"text": message['content']}]})
    
        # Initialize the Gemini model with the system instruction
        genai.configure(api_key=google_api_key)        
        genai_model = genai.GenerativeModel(model, system_instruction=system_prompt)
    
        response = genai_model.generate_content(
            gemini_content,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.95,
                top_k=40,
                stop_sequences=stop_sequences
            )
        )
        prompt_tokens = genai_model.count_tokens(gemini_content).total_tokens
        completion_tokens = genai_model.count_tokens(response.text).total_tokens
        total_tokens = prompt_tokens + completion_tokens
    
        return_data = {
            "choices": [{
                "message": {
                    "content": response.text,
                    "role": "assistant"
                }
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }
    
        return return_data
        
    elif model.startswith('gpt-') or model.startswith('o1-'):
        openai_api_key = get_api_key(openai_api_key=openai_api_key)
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",
        }
        data = {
            "model": model,
            "messages": messages,
        }
        
        # Special case for 'o1' and 'o1-mini' models
        if model.startswith('o1-'):
            data["max_completion_tokens"] = max_tokens
            # Omit unsupported parameters for 'o1' models
        else:
            data["max_tokens"] = max_tokens
            data["temperature"] = temperature
            data["n"] = n
            # Only include functions, function_call, logit_bias and stop_sequences in the data dict if they are not empty or None.
            if functions is not None:
                data["functions"] = functions
            if function_call is not None:
                data["function_call"] = function_call
            if logit_bias is not None:
                data["logit_bias"] = logit_bias
            if stop_sequences is not None:
                data["stop"] = stop_sequences  # GPT calls it 'stop' in the settings dictionary, even though they call it stop_sequences in the docs
    
        if system_prompt is not None:
            if messages and messages[0]["content"] != system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
    
    elif model.startswith('claude-'):
        anthropic_api_key = get_api_key(anthropic_api_key=anthropic_api_key)
        
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": anthropic_api_key,
            "anthropic-version": "2023-06-01"
        }
        data = {
            "model": model,
            "messages": gpt_to_anthropic_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if stop_sequences is not None:
            data["stop_sequences"] = stop_sequences
        if system_prompt is not None:
            data["system"] = system_prompt
    else:
        raise ValueError(f"Invalid model in send_llm_request: {model}")
    
    safe_headers = headers.copy()
    if "Authorization" in safe_headers:
        safe_headers["Authorization"] = "Bearer ********"
    if "x-api-key" in safe_headers:
        safe_headers["x-api-key"] = "sk-********"
    logger.debug(f"Post to: {url}\nHeaders: {safe_headers}\nData: {json.dumps(data, indent=4)}")
    response = requests.post(url, headers=headers, json=data, timeout=timeout)
    logger.debug(f"Response: {json.dumps(response.json(), indent=4)}")
    
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logger.warning(f"Error: {e}; Response: {response.text}")
        raise
    return response.json()

def handle_error(error: Exception, model: str) -> bool:
    gpt_error_messages = {
        (ValueError,): "ValueError encountered. Aborting. Check the request parameters.",
        (FunctionTimedOut, ConnectTimeout, ReadTimeout): "Timeout error. Retrying after a brief wait.",
        openai.APIConnectionError: "API Connection error. Retrying after a brief wait.",
        openai.APITimeoutError: "API Timeout error. Retrying after a brief wait.",
        openai.AuthenticationError: "Authentication error. Aborting. Check the API key or token.",
        openai.BadRequestError: "Bad Request error. Aborting. Check the request parameters.",
        openai.ConflictError: "Conflict error. Retrying after a brief wait.",
        openai.InternalServerError: "Internal Server error. Retrying after a brief wait.",
        openai.NotFoundError: "Not Found error. Aborting. Check the resource ID.",
        openai.PermissionDeniedError: "Permission Denied error. Aborting. Check the API key, organization ID, and resource ID.",
        openai.RateLimitError: "Rate Limit error. Retrying after a brief wait.",
        openai.UnprocessableEntityError: "Unprocessable Entity error. Please try the request again."
    }
    claude_error_messages = {
        (ValueError,): "ValueError encountered. Aborting. Check the request parameters.",
        (FunctionTimedOut, ConnectTimeout, ReadTimeout): "Timeout error. Retrying after a brief wait.",
        requests.exceptions.HTTPError: "HTTP error. Retrying after a brief wait.",
        requests.exceptions.ConnectionError: "Connection error. Retrying after a brief wait.",
        requests.exceptions.Timeout: "Timeout error. Retrying after a brief wait.",
        requests.exceptions.TooManyRedirects: "Too Many Redirects error. Aborting. Check the request URL.",
        requests.exceptions.RequestException: "Request Exception. Aborting. Check the request."
    }
    gemini_error_messages = {
        (ValueError,): "ValueError encountered. Aborting. Check the request parameters.",
        (FunctionTimedOut, ConnectTimeout, ReadTimeout): "Timeout error. Retrying after a brief wait.",
        google.api_core.exceptions.InvalidArgument: "Invalid Argument error. Aborting. Check the request parameters.",
        google.api_core.exceptions.PermissionDenied: "Permission Denied error. Aborting. Check the API key or permissions.",
        google.api_core.exceptions.ResourceExhausted: "Resource Exhausted error. Retrying after a brief wait.",
        google.api_core.exceptions.NotFound: "Not Found error. Aborting. Check the resource ID.",
        google.api_core.exceptions.DeadlineExceeded: "Deadline Exceeded error. Retrying after a brief wait.",
        google.api_core.exceptions.ServiceUnavailable: "Service Unavailable error. Retrying after a brief wait.",
        google.api_core.exceptions.InternalServerError: "Internal Server error. Retrying after a brief wait.",
        google.auth.exceptions.DefaultCredentialsError: "Default Credentials error. Aborting. Check the API key or authentication.",
        google.api_core.exceptions.RetryError: "Retry error. Retrying after a brief wait.",
        google.generativeai.types.BlockedPromptException: "Blocked Prompt error. Aborting. The prompt was blocked due to safety reasons.",
        google.generativeai.types.StopCandidateException: "Stop Candidate error. Aborting. The API responded with an exceptional finish_reason.",
    }
    
    if model.startswith('gpt-') or model.startswith('o1-'):
        error_messages = gpt_error_messages
    elif model.startswith('claude-'):
        error_messages = claude_error_messages
    elif model.startswith('gemini-'):
        error_messages = gemini_error_messages
    else:
        raise ValueError(f"Unknown model category in handle_error: {model}")

    # Special handling for HTTPError with specific status codes
    if isinstance(error, requests.exceptions.HTTPError):
        response_text = getattr(error.response, 'text', '')
        status_code = error.response.status_code
        # Don't retry if content filtering blocked the output
        if "Output blocked by content filtering policy" in response_text:
            logger.error(f"Content filtering policy error using {model}: {str(error)}. Aborting.")
            return True
        elif status_code == 400:
            logger.error(f"{status_code} Bad Request error. Aborting. Check the request parameters.")
            return True
        if status_code == 401:
            logger.error(f"{status_code} Unauthorized error. Aborting. Check your API key.")
            return True
        elif status_code == 404:
            logger.error(f"{status_code} Not Found error. Aborting. Check the resource ID.")
            return True

    for error_type, message in error_messages.items():
        if isinstance(error, error_type):
            if "Aborting" in message:
                logger.error(f"{type(error).__name__} using {model}: {str(error)}. {message}")
                return True
            elif "Retrying" in message:
                logger.warning(f"{type(error).__name__} using {model}: {str(error)}. {message}")
            else:
                logger.error(f"{type(error).__name__} using {model}: {str(error)}. {message}")
            return False
    logger.error(f"Unhandled {type(error).__name__} using {model}: {str(error)}. Retrying after a brief wait.")
    return False

def gpt_to_anthropic_messages(gpt_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    anthropic_messages = []
    for message in gpt_messages:
        if message['role'] == 'user':
            anthropic_messages.append({"role": "user", "content": message['content']})
        elif message['role'] == 'assistant':
            anthropic_messages.append({"role": "assistant", "content": message['content']})
        elif message['role'] == 'system':
            # Skip system messages as they are handled separately
            continue
        else:
            logger.error(f"Unexpected message role: {message['role']}. Appending it as a user message.")
            anthropic_messages.append({"role": "user", "content": message['content']})
    return anthropic_messages

def send_llm_request_with_retries(settings: Dict[str, Any], messages: List[Dict[str, str]], system_prompt: Optional[str] = None, openai_api_key: Optional[str] = None, anthropic_api_key: Optional[str] = None, google_api_key: Optional[str] = None) -> Optional[dict]:    
    """
    Calls send_llm_request adding retries in case of errors.
    Called by call_gpt, which prepares the settings and messages.
    """
    if messages is None:
        raise ValueError("messages cannot be None")
    model = settings["model"]
    msg_tokens = num_tokens_from_messages(messages, model, settings.get("functions", None), settings.get("function_call", None), system_prompt, anthropic_api_key)
    available_tokens = get_model_input_token_max(model) - msg_tokens - 1
    if available_tokens < 1:
        logger.error(f"Messages[] has a length of {msg_tokens}. Max tokens available is less than 1: {available_tokens}")
        return None
    if available_tokens < settings["min_reply_tokens"]:
        logger.error(f'Messages[] has a length of {msg_tokens}. Available tokens {available_tokens} is less than the desired minimum reply length of {settings["min_reply_tokens"]}.')
        return None
    local_max_tokens = settings["max_tokens"]
    if settings["max_tokens"] is None:
        local_max_tokens = available_tokens
    if settings["max_tokens"] > available_tokens:
        local_max_tokens = available_tokens
    logger.debug(f'Messages[] has a length of {msg_tokens}. Reply can be {local_max_tokens} tokens of {available_tokens} available.')

    max_attempts = 6
    backoff_time = 3  # time in seconds
    circuit_breaker = False
    timeout = max(msg_tokens/2, 300)  # seconds
    last_error = None
    api_start_time = time.perf_counter()

    for attempt in range(max_attempts):
        api_start_time = time.perf_counter()
        try:
            response = send_llm_request(
                model=model,
                temperature=settings.get("temperature", 0),
                messages=messages,
                max_tokens=local_max_tokens,
                n=settings.get("n", 1),
                timeout=timeout,
                logit_bias=settings.get("logit_bias", None),
                functions=settings.get("functions", None),
                function_call=settings.get("function_call", None),
                stop_sequences=settings.get("stop_sequences", None),
                system_prompt=system_prompt,
                openai_api_key=openai_api_key,
                anthropic_api_key=anthropic_api_key,
                google_api_key=google_api_key
            )
            logger.debug(f"Successful API call. Retries needed: {attempt}. model: {model}.")
            return response
        except Exception as e:
            last_error = e
            logger.error(f"Error: {e}")
            circuit_breaker = handle_error(e, model)
            if circuit_breaker:
                break
            time_elapsed = time.perf_counter() - api_start_time
            logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed after {time_elapsed:.1f} seconds. Retrying in {backoff_time} seconds...")
            if attempt < max_attempts - 1:
                time.sleep(backoff_time * random.uniform(1,2))
                backoff_time *= 2
    if circuit_breaker:
        logger.error(f"Circuit breaker triggered. Aborting.")
        raise last_error
    logger.error(f"Maximum number of attempts ({max_attempts}) reached, last error was {last_error}, giving up.")
    raise last_error

DEBUG = globals().get("DEBUG", False)

def call_gpt(
    query: Union[str, List[Dict[str, str]]],
    settings: Optional[Dict[str, Any]] = None,
    return_pricing: bool = False,
    return_dict: bool = False,
    functions: Optional[List[dict]] = None,
    function_call: Optional[dict] = None,
    logit_bias: Optional[dict] = None,
    stop_sequences: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None
) -> Union[str, Dict[str, Any], Tuple[str, float], Tuple[Dict[str, Any], float]]:
    """
    This function prepares the settings and messages for the API call and then calls send_llm_request_with_retries.
    It handles the response from the API and logs the result.
    """
    start_time = time.perf_counter()
    logger.debug(f"Sending GPT API query with settings {settings}, called from " + str(inspect.currentframe().f_back.f_code.co_name))
    start_time = time.perf_counter()

    if settings is None:
        local_settings = {}
    else:
        local_settings = settings.copy()  # Create a local copy of the settings dictionary
        local_settings["model"] = expand_model_shortcut(local_settings["model"])
    local_settings.setdefault("model", "gpt-4-turbo")
    local_settings.setdefault("temperature", 0.7)
    local_settings.setdefault("n", 1)
    local_settings.setdefault("max_tokens", get_model_output_token_max(local_settings["model"]))
    local_settings.setdefault("min_reply_tokens", min(get_model_input_token_max(local_settings["model"]) // 4, get_model_output_token_max(local_settings["model"])))

    if functions is not None:
        local_settings["functions"] = functions
    if function_call is not None:
        local_settings["function_call"] = function_call
    if logit_bias is not None:
        local_settings["logit_bias"] = logit_bias
    if stop_sequences is not None:
        local_settings["stop_sequences"] = stop_sequences

    # Create a message array from a query if needed
    if isinstance(query, str):
        if query == "":
            raise ValueError("Blank query string")
        messages = [{"role": "user", "content": query}]
    elif isinstance(query, list):
        messages = query
        # Make sure messages isn't too big, and if so, trim it
        messages = trim_messages(
            messages,
            local_settings["model"],
            local_settings["min_reply_tokens"],
            local_settings.get("functions", None),
            local_settings.get("function_call", None),
            system_prompt,
            local_settings.get("anthropic_api_key", None)
        )
    else:
        raise ValueError("query is wrong type - must be a string or a message list")
    if not messages:
        raise ValueError("must submit either a string or a message list")
    msg_tokens = num_tokens_from_messages(
        messages,
        local_settings["model"],
        local_settings.get("functions", None),
        local_settings.get("function_call", None),
        system_prompt,
        anthropic_api_key
    )
    logger.debug(f"Sending LLM API query, {msg_tokens} tokens:\n{messages}")
    try:
        response = send_llm_request_with_retries(
            local_settings,
            messages,
            system_prompt,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            google_api_key=google_api_key
        )
    except Exception as e:
        logger.error(f"GPT API query failed with error: {e}")
        raise e
    logger.debug(f"completed api query at {round(time.perf_counter() - start_time)} seconds.")
    reply = ""
    if response is None:
        logger.error(f"API reply is None")
    else:
        if local_settings["model"].startswith('claude-'):
            reply = response["content"][-1]["text"] if response["content"] and response["content"][-1]["type"] == "text" else ""
        else: 
            reply = response['choices'][0]['message']['content'] or ""
    total_cost = get_cost(
        local_settings["model"],
        num_tokens_from_messages(
            messages,
            local_settings["model"],
            local_settings.get("functions", None),
            local_settings.get("function_call", None),
            system_prompt,
            anthropic_api_key
        ),
        get_tokens(reply, local_settings["model"], anthropic_api_key)
    )
    logger.debug(f"Reply is {get_tokens(reply, local_settings['model'], anthropic_api_key)} tokens using {local_settings['model']} with a total cost of ${total_cost}: \n{reply}")
    if return_pricing:
        if return_dict:
            return response, total_cost
        else:
            return reply, total_cost
    if return_dict:
        return response
    else:
        return reply

# Ask a yes or no question
def ask_yes_or_no(prompt: str, settings: Optional[Dict[str, Any]] = None, openai_api_key: Optional[str] = None, anthropic_api_key: Optional[str] = None, google_api_key: Optional[str] = None) -> str:
    # Function to get a one-char 'y' or 'n' response using logit bias.
    if settings is None:
        settings = {"model": "gpt-4-turbo"}
    settings["temperature"] = 0  # override these no matter what
    settings["max_tokens"] = 1

    if settings["model"].startswith("gpt-"):
        settings.update({
            "logit_bias": {88: 100, 77: 100}  # 'y' and 'n' are represented by tokens 88 and 77
        })
        response = call_gpt(
            [ { "role": "user", "content": f"Answer the following question with one character, either y or n:\n{prompt}" } ],
            settings=settings,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            google_api_key=google_api_key
        )
        return response.lower()

    else:
        response = call_gpt(
            [
                {"role": "user", "content": f"Answer the following question with one character in JSON format, either y or n:\n{prompt}"},
                {"role": "assistant", "content": '{{"answer": "'}  # Give it all the JSON starter so it just gives you the y or n
            ],
            settings=settings,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            google_api_key=google_api_key
        )
        return response.lower()

def main():
    # Prompt user for query string
#    query = input("Enter your query > ")
    query = "Please recite the alphabet in ALL CAPS."

    # Call GPT API
    settings = {"model": "gpt-4", "temperature": 0.0, "max_tokens": 100}
    try:
        response = call_gpt(query, settings=settings)
    except Exception as e:
        print("GPT API call failed with error: ", e)
    else:
        print("GPT response: ", response)

    # Call Anthropic API
    settings = {"model": "claude", "temperature": 0.0, "max_tokens": 100}
    try:
        response = call_gpt(query, settings=settings)
    except Exception as e:
        print("Anthropic API call failed with error: ", e)
    else:
        print("Anthropic response: ", response)

    # Call Gemini API
    settings = {"model": "gemini", "temperature": 0.0, "max_tokens": 100}
    try:
        response = call_gpt(query, settings=settings)
    except Exception as e:
        print("Gemini API call failed with error: ", e)
    else:
        print("Gemini response: ", response)

    # Run tests
    print("Now running tests...")
    loader = unittest.defaultTestLoader
    pattern = f"{os.path.splitext(os.path.basename(__file__))[0]}_tests.py"
    print(f"Looking for test files matching pattern: {pattern} in directory: {os.path.abspath('.')}")
    suite = loader.discover(start_dir='.', pattern=pattern)

    print("Discovered test files:")
    for test_suite in suite:
        for test_case in test_suite:
            test_class = test_case.__class__
            module_name = test_class.__module__
            module = __import__(module_name)
            file_path = os.path.abspath(module.__file__)
            print(f"  - {file_path}")

    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise ValueError("Some of the tests failed, aborting.")
    else:
        print("All tests passed!")

if __name__ == "__main__":
    main()