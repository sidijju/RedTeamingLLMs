import os
import json
import time
from openai import OpenAI
from typing import Union, List, Dict
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer 
from anthropic import Anthropic

# Load credentials from creds.env
if not load_dotenv("creds.env"):
    print("Error: creds.env file not found")
    exit(1)

def get_env_variable(var_name):
    """Fetch environment variable or return None if not set."""
    return os.getenv(var_name)

CALL_SLEEP = 1
clients = {}

def initialize_clients():
    """Dynamically initialize available clients based on environment variables."""
    try:
        # GPT client setup
        gpt_api_key = get_env_variable('GPT_API_KEY')
        gpt_base_url = get_env_variable('BASE_URL_GPT')
        if gpt_api_key and gpt_base_url:
            clients['gpt'] = OpenAI(base_url=gpt_base_url, api_key=gpt_api_key)

        # Claude client setup
        claude_api_key = get_env_variable('CLAUDE_API_KEY')
        claude_base_url = get_env_variable('BASE_URL_CLAUDE')
        if claude_api_key:
            clients['claude'] = Anthropic(api_key = claude_api_key)

        # DeepInfra client setup
        deepinfra_api_key = get_env_variable('DEEPINFRA_API_KEY')
        deepinfra_base_url = get_env_variable('BASE_URL_DEEPINFRA')
        if deepinfra_api_key and deepinfra_base_url:
            clients['deepinfra'] = OpenAI(base_url=deepinfra_base_url, api_key=deepinfra_api_key)

        if not clients:
            print("No valid API credentials or models found. Exiting.")
            exit(1)

    except Exception as e:
        print(f"Error during client initialization: {e}")
        exit(1)

initialize_clients() 


def get_client(model_name):
    """Select appropriate client based on the given model name."""
    if 'gpt' in model_name or 'o1-' in model_name:
        client = clients.get('gpt') 
    elif 'claude' in model_name:
        client = clients.get('claude')
    elif 'deepseek' in model_name:
        client = clients.get('deepseek')
    else:
        client = clients.get('deepinfra')

    if not client:
        raise ValueError(f"{model_name} client is not available or unknown.")
    return client


def read_prompt_from_file(filename):
    with open(filename, 'r') as file:
        prompt = file.read()
    return prompt

def read_data_from_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def parse_json(output):
    try:
        output = ''.join(output.splitlines())
        if '{' in output and '}' in output:
            start = output.index('{')
            end = output.rindex('}')
            output = output[start:end + 1]
        data = json.loads(output)
        return data
    except Exception as e:
        print("parse_json:", e)
        print("Offending JSON string:", output)  # Debugging print to show exact content

        return None
    
def check_file(file_path):
    if os.path.exists(file_path):
        return file_path
    else:
        raise IOError(f"File not found error: {file_path}.")

def gpt_call(client, query: Union[List, str], model_name="gpt-4o", temperature=0):
    if isinstance(query, list) and 'claude' in model_name:
        messages = [{"role": m["role"], "content": m["content"]} for m in query]
    else:
        if isinstance(query, list) and all(isinstance(item, dict) and "content" in item for item in query):  # For Hugging Face models, ensure query is a single string

            query = " ".join(item["content"] for item in query)
        messages = query if isinstance(query, list) else [{"role": "user", "content": query}]
    # For Hugging Face models (e.g., CodeGen)
    if isinstance(client, dict) and "model" in client and "tokenizer" in client:
        model = client['model']
        tokenizer = client['tokenizer']
        
        # Set pad_token to eos_token if pad_token is not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize input with attention_mask
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Use `max_new_tokens` instead of `max_length`, and set `pad_token_id` for reliable padding
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,  # Limit the number of tokens generated
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    
    for _ in range(3):
        try:
            if "claude" in model_name:
                completion = client.messages.create(
                    model=model_name,  # IF BUGGY, ENSURE NAME IS VALID
                    max_tokens=8192,
                    messages=messages
                )
                resp = completion.content[0].text
            else:
                # All OpenAI-compatible APIs (GPT, DeepInfra, etc)
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages
                )
                resp = completion.choices[0].message.content

            return resp
        
        except Exception as e:
            print(f"GPT_CALL Error: {model_name}:{e}")
            time.sleep(CALL_SLEEP)
            continue
    return ""


def gpt_call_append(client, model_name, dialog_hist, query):
    """Append a new message to dialog history and get response"""
    if isinstance(query, str) or "claude" in model_name:
        dialog_hist.append({"role": "user", "content": query})
    else:
        dialog_hist.append(query)
    
    try:
        if "claude" in model_name:

            # Claude API format
            response = client.messages.create(
                model=model_name,
                max_tokens=8192,
                messages=[{
                        'role': m['role'],
                        'content': m['content']
                    } for m in dialog_hist] 
            )
            resp = response.content[0].text
        else:
            # OpenAI API format
            response = client.chat.completions.create(
                model=model_name,
                messages=dialog_hist
            )
            resp = response.choices[0].message.content
            
        dialog_hist.append({"role": "assistant", "content": resp})
        return resp, dialog_hist
        
    except Exception as e:
        print(f"Error in gpt_call_append: {e}")
        raise e
