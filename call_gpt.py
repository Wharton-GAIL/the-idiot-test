"""
# call_gpt documentation

call_gpt sends a query to an LLM and returns a response. Syntax:
###def call_gpt(query: Union[str, List[Dict[str, str]]], settings: Optional[Dict[str, Any]] = None, return_pricing: bool = False, return_dict: bool = False, functions: Optional[List[dict]] = None, function_call: Optional[dict] = None, logit_bias: Optional[dict] = None, stop_sequences: Optional[List[str]] = None, system_prompt: Optional[str] = None) -> Union[str, Dict[str, Any], Tuple[str, float], Tuple[Dict[str, Any], float]]:

**Parameters:**
- `query`: a string will be converted to a single user message. 
  If it is a list, format like this:
    [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Name a color."},
        {"role": "assistant", "content": "Red"}
        {"role": "user", "content": "Another?"},
    ]
- `settings` if omitted, sensible defaults will be used. If included, overrides the defaults for some or all of these keys: 
* "model" can be `gpt-3.5-turbo` or `gpt-4-turbo` or others
* "temperature" default 0.7
* "n" number of replies
* "max_tokens" default is the max allowable
* "min_reply_tokens" default is max_tokens/4
* "openai_api_key" required for OpenAI API authentication
* "anthropic_api_key" required for Anthropic API authentication
- `return_pricing`: If True, returns the cost of the call as a float representing USD
- `return_dict`: If True, returns the full API response as a dictionary; otherwise, just the reply string. 
- `functions`: List of dicts specifying function(s) the LLM can call.
    "name": Name of the function.
    "parameters": Object detailing function's arguments and their types.
- function_call: Dict to invoke a function from the API call.
    "name": Name of the function to call.
    "args": Arguments for the function.
- `logit_bias`: Token IDs from tiktoken are keys. Biases from -100 to 100 are valid. 
- `stop_sequences`: List of strings that will halt LLM output if generated, without generating them
- `system_prompt`: Optional system prompt to set the behavior of the model.

**Returns:**
- By default, returns reply content as a string.
- If `return_dict` is True, returns full API response as a dict.
- If `return_pricing` is True, returns a tuple with the first element being the reply (string or dict) and the second being the cost."""

import openai
import tiktoken
import google.generativeai as genai # pip install google-generativeai
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

import os, sys
if os.name == 'nt':  # Windows
    sys.path.append('..\\dan-tools')
else:  # Linux
    sys.path.append('../dan-tools')
from log_love import setup_logging

logger = None
logger = setup_logging(logger)

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
    logger.error("Update to call get_model_input_token_max or get_model_output_token_max")
    return get_model_output_token_max(model)

def get_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    input_price_per_token, output_price_per_token = (price / 1000000 for price in MODELS[model]["pricing"])
    input_cost = input_price_per_token * input_tokens
    output_cost = output_price_per_token * output_tokens
    total_cost = input_cost + output_cost
    return total_cost

def get_tokens(to_tokenize: str, model: str) -> Optional[int]:
    model = expand_model_shortcut(model)
    if not isinstance(to_tokenize, str):
        raise ValueError("Input to tokenize must be a string")
    if model.startswith('gemini-'):
        genai_model = genai.GenerativeModel(model)
        return genai_model.count_tokens(to_tokenize).total_tokens
    elif model.startswith('claude-'):
        return anthropic_client.count_tokens(to_tokenize)
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

def num_tokens_from_messages(messages: List[Dict[str, str]], model: str, functions: Optional[List[dict]] = None, function_call: Optional[dict] = None, system_prompt: Optional[str] = None) -> int:
    if model.startswith('gemini-'):
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
                tokens = get_tokens(value, model)
                if tokens is None:
                    logger.error("Tokenization failed for the value: {value}")
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
            num_tokens += get_tokens(str(function), model)  # for the function name

    if function_call:
        num_tokens += get_tokens(str(function_call), model)  # for the function call

    if system_prompt:
        num_tokens += get_tokens(system_prompt, model)
    
    return num_tokens

def trim_messages(messages: List[Dict[str, str]], model: str, min_reply_tokens: Optional[int] = None, functions: Optional[List[dict]] = None, function_call: Optional[dict] = None, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    # Reserve for output 1/4 of input tokens or all of output tokens, whichever is smaller
    if min_reply_tokens is None:
        min_reply_tokens = min(get_model_input_token_max(model) / 4, get_model_output_token_max(model))
    messages_tokens = num_tokens_from_messages(messages, model, functions, function_call, system_prompt)
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
        messages_tokens = num_tokens_from_messages(trimmed_messages, model, functions, function_call, system_prompt)
    if messages_deleted > 0:
        logger.info(
            f"TRIMMED: {messages_deleted} messages deleted. Required {min_reply_tokens} tokens for reply in {model} ({get_model_output_token_max(model)} output token max), so messages trimmed from {start_size} to {messages_tokens} tokens. Output can have {get_model_output_token_max(model)-messages_tokens} tokens now."
        )
    else:
        logger.debug(
            f"No messages deleted. Required {min_reply_tokens} tokens for reply in {model} ({get_model_output_token_max(model)} output token max), so messages trimmed from {start_size} to {messages_tokens} tokens. Output can have {get_model_output_token_max(model)-messages_tokens} tokens now."
        )
    return trimmed_messages

def gpt_to_anthropic_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Convert GPT-style messages to Anthropic format."""
    anthropic_messages = []
    for message in messages:
        if message["role"] == "user":
            anthropic_messages.append({"role": "user", "content": message["content"]})
        elif message["role"] == "assistant":
            anthropic_messages.append({"role": "assistant", "content": message["content"]})
        elif message["role"] == "system":
            # Skip system messages as they're handled separately in Anthropic's API
            continue
        else:
            logger.warning(f"Unexpected message role: {message['role']}")
    return anthropic_messages

def handle_error(e: Exception, model: str) -> bool:
    """Handle API errors and determine if we should break the retry loop."""
    error_message = str(e).lower()
    
    # Check for rate limit errors
    if any(phrase in error_message for phrase in ["rate limit", "ratelimit", "too many requests", "429"]):
        logger.warning(f"Rate limit hit for {model}. Waiting before retry...")
        return False
    
    # Check for timeout errors
    if isinstance(e, (ReadTimeout, ConnectTimeout, FunctionTimedOut)) or "timeout" in error_message:
        logger.warning(f"Timeout error for {model}. Will retry...")
        return False
    
    # Check for temporary server errors
    if any(phrase in error_message for phrase in ["server error", "500", "502", "503", "504"]):
        logger.warning(f"Server error for {model}. Will retry...")
        return False
    
    # Check for authentication errors
    if any(phrase in error_message for phrase in ["authentication", "auth", "unauthorized", "401"]):
        logger.error(f"Authentication error for {model}. Check your API key.")
        return True
    
    # Check for invalid requests
    if any(phrase in error_message for phrase in ["invalid request", "bad request", "400"]):
        logger.error(f"Invalid request for {model}. Check your parameters.")
        return True
    
    # Check for model-specific errors
    if "model" in error_message and any(phrase in error_message for phrase in ["not found", "doesn't exist", "unavailable"]):
        logger.error(f"Model error for {model}. Check if the model is available.")
        return True
    
    # Check for content policy violations
    if any(phrase in error_message for phrase in ["content policy", "content filter", "content violation"]):
        logger.error(f"Content policy violation for {model}.")
        return True
    
    # Default case: log the error and continue retrying
    logger.warning(f"Unhandled error for {model}: {error_message}")
    return False

# Initialize Anthropic client
anthropic_client = None

def init_anthropic(anthropic_api_key: str):
    global anthropic_client
    if anthropic_client is None or anthropic_client.api_key != anthropic_api_key:
        anthropic_client = Anthropic(api_key=anthropic_api_key)
    return anthropic_client

def send_llm_request(
    model: str,
    temperature: float,
    messages: List[dict],
    max_tokens: int,
    n: int,
    timeout: float,
    openai_api_key: str,
    anthropic_api_key: str,
    stop_sequences: Optional[List[str]] = None,
    logit_bias: Optional[dict] = None,
    functions: Optional[List[dict]] = None,
    function_call: Optional[dict] = None,
    system_prompt: Optional[str] = None
) -> dict:
    """
    Sends a request to the LLM API.
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
                # Handle system messages or any other roles as user messages
                gemini_content.append({"role": "user", "parts": [{"text": message['content']}]})

        # Initialize the Gemini model with the system instruction
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
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",  # Use the OpenAI API key
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
            # Only include functions, function_call, logit_bias and stop_sequences if they are provided
            if functions is not None:
                data["functions"] = functions
            if function_call is not None:
                data["function_call"] = function_call
            if logit_bias is not None:
                data["logit_bias"] = logit_bias
            if stop_sequences is not None:
                data["stop"] = stop_sequences  # GPT calls it 'stop' in the settings

        if system_prompt is not None:
            if messages and messages[0]["content"] != system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

    elif model.startswith('claude-'):
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

def send_llm_request_with_retries(settings: Dict[str, Any], messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> Optional[dict]:    
    """
    Calls send_llm_request adding retries in case of errors.
    Called by call_gpt, which prepares the settings and messages.
    """
    if messages is None:
        raise ValueError("messages cannot be None")
    model = settings["model"]
    msg_tokens = num_tokens_from_messages(messages, model, settings.get("functions", None), settings.get("function_call", None), system_prompt)
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
                openai_api_key=settings.get("openai_api_key"),  # Pass the OpenAI API key
                anthropic_api_key=settings.get("anthropic_api_key"),  # Pass the Anthropic API key
                logit_bias=settings.get("logit_bias", None),
                functions=settings.get("functions", None),
                function_call=settings.get("function_call", None),
                stop_sequences=settings.get("stop_sequences", None),
                system_prompt=system_prompt
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

def call_gpt(
    query: Union[str, List[Dict[str, str]]], 
    settings: Optional[Dict[str, Any]] = None, 
    return_pricing: bool = False, 
    return_dict: bool = False, 
    functions: Optional[List[dict]] = None, 
    function_call: Optional[dict] = None, 
    logit_bias: Optional[dict] = None, 
    stop_sequences: Optional[List[str]] = None, 
    system_prompt: Optional[str] = None
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
            system_prompt
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
        system_prompt
    )
    logger.debug(f"Sending LLM API query, {msg_tokens} tokens:\n{messages}")
    try:
        response = send_llm_request_with_retries(
            local_settings, 
            messages, 
            system_prompt
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
            system_prompt
        ), 
        get_tokens(reply, local_settings["model"])
    )
    logger.debug(f"Reply is {get_tokens(reply, local_settings['model'])} tokens using {local_settings['model']} with a total cost of ${total_cost}: \n{reply}")
    if return_pricing:
        if return_dict:
            return response, total_cost
        else:
            return reply, total_cost
    if return_dict:
        return response
    else:
        return reply

if __name__ == "__main__":
    # Example usage
    settings = {
        "model": "gpt-4-turbo",
        "temperature": 0.7,
        "openai_api_key": "your-openai-api-key-here",  # Add your OpenAI API key here
        "anthropic_api_key": "your-anthropic-api-key-here"  # Add your Anthropic API key here
    }
    
    # Test basic query
    response = call_gpt("Hello, how are you?", settings)
    print(f"Basic response: {response}")
    
    # Test with system prompt
    response = call_gpt(
        "What color is the sky?",
        settings,
        system_prompt="You are a meteorologist. Be scientific in your responses."
    )
    print(f"Response with system prompt: {response}")
    
    # Test with pricing
    response, cost = call_gpt(
        "Tell me a short story.",
        settings,
        return_pricing=True
    )
    print(f"Response with cost: {response}\nCost: ${cost}")
    
    # Test with message history
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's your favorite color?"},
        {"role": "assistant", "content": "I don't have personal preferences, but I can discuss colors!"},
        {"role": "user", "content": "Tell me about that color."}
    ]
    response = call_gpt(messages, settings)
    print(f"Response with message history: {response}")