import os
import openai
import re
import statistics
import time
from messages import *
# https://callmefred.com/how-to-fix-openai-error-ratelimiterror-the-server-had-an-error/
MODEL = "gpt-3.5-turbo"
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(model=MODEL, prompt=""):
    while True:
        try:
            response = openai.ChatCompletion.create(
            model=model,
            messages=prompt,
            temperature=0,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
            break
        except openai.error.RateLimitError as e:
            retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
            print(f"\nRate limit exceeded. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            continue
        except openai.error.Timeout as e:
            print(f"\nRequest timed out: {e}. Retrying in 10 seconds...")
            time.sleep(10)
            continue
        except openai.error.APIError as e:
            retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
            print(f"\nAPI error occurred. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            continue
        except openai.error.ServiceUnavailableError as e:
            print(f"\nService is unavailable. Retrying in 10 seconds...")
            time.sleep(10)
            continue
    return response


def get_prediction(sentence: str = "", prompt: str = "", model = MODEL):
    if len(sentence) == 0:
        return 0.5
    prompt.append({"role": "user", "content": f"<review>{sentence}<review>"})
    to_sum = []
    response = generate_response(model,prompt)
    outcome = response["choices"][0]["message"]["content"]
    outcome = eval(outcome)
    prediction = outcome[1] if outcome[0] == 1 else (1-outcome[1])
    prompt.pop()
    return float(prediction)