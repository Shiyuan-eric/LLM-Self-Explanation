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
            print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            continue
        except openai.error.Timeout as e:
            print(f"Request timed out: {e}. Retrying in 10 seconds...")
            time.sleep(10)
            continue
        except openai.error.APIError as e:
            retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
            print(f"API error occurred. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            continue
        except openai.error.ServiceUnavailableError as e:
            print(f"Service is unavailable. Retrying in 10 seconds...")
            time.sleep(10)
            continue
    return response


def get_prediction(sentence: str = "", prompt: str = "", model = MODEL):
    if len(sentence) == 0:
        return (0.5, False)
    prompt.append({"role": "user", "content": f"\"{sentence}\""})
    to_sum = []
    print(prompt)
    response = generate_response(model,prompt)
    outcome = response["choices"][0]["message"]["content"]
    print(outcome)
    m = re.search("[-+]?(?:\d*\.*\d+)", outcome)
    if m:
        to_sum.append(float(m[0]))
    prompt.pop()
    # mean = sum(to_sum) / len(to_sum)
    # variance = sum([((x - mean) ** 2) for x in to_sum]) / len(to_sum)
    # res = variance ** 0.5
    # print(res)
    # print(to_sum)

    if len(to_sum) == 0:
        print(f"The sentence is: {sentence}")
        return (0.5, True)
    return (to_sum[0], False)