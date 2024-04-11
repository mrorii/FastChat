# Adapted from https://github.com/Libr-AI/do-not-answer/blob/30ae028/do_not_answer/utils/openai_api.py
import os
import time
import openai


def setup_openai_api(model: str, use_azure=True):
    from functools import partial

    if model == "gpt-3.5-turbo":
        deployment_id = "misc-35"
    elif model == "gpt-4":
        deployment_id = "misc-4"
    else:
        raise NotImplementedError(f"{model=}")

    if use_azure:
        openai.api_type = "azure"
        openai.api_key = os.environ['OPENAI_AZURE_API_KEY']
        openai.api_base = os.environ['OPENAI_AZURE_API_BASE']
        openai.api_version = "2023-05-15"  # subject to change
        return partial(openai.ChatCompletion.create, deployment_id=deployment_id)
    else:
        return openai.ChatCompletion.create


def gpt_single_try(messages, model="gpt-4", max_tokens=None):
    openai_chat_completion_func = setup_openai_api(model)
    response = openai_chat_completion_func(
        model=model,
        messages=messages,
        max_tokens=max_tokens)

    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result

def gpt(messages, model = "gpt-3.5-turbo", num_retries=3, max_tokens=None):
    r = ''
    for _ in range(num_retries):
        try:
            r = gpt_single_try(messages, model, max_tokens=max_tokens)
            break
        except Exception as e:
            print(type(e), e)
            if ("content management policy" in str(e)) or (
                    "maximum context length is 4096 tokens. However, you requested" in str(e)) or (
                    "Detected an error in the prompt" in str(e)) or ("Sorry! We've encountered" in str(e)):
                return "Rating: [[0]]"
            time.sleep(6)
    return r


def chatgpt_single_try(user_input, system_role = "You are a helpful assistant."):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_input},
        ]
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result


def chatgpt(user_input, system_role="You are a helpful assistant.", num_retries=3):
    r = ''
    for _ in range(num_retries):
        try:
            r = chatgpt_single_try(user_input, system_role)
            break
        except openai.error.OpenAIError as exception:
            print(f"{exception}. Retrying...")
            time.sleep(1)
    return r
