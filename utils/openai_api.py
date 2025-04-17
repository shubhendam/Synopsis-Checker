# utils/openai_api.py

import os
import openai
from dotenv import load_dotenv

#load env variables from .env
load_dotenv()

#get api key
openai.api_key = os.getenv("OPENAI_API_KEY")

def call_chatgpt(prompt, model="gpt-3.5-turbo", max_tokens=512):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful evaluator AI."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )

        return response['choices'][0]['message']['content'].strip()

    except Exception as e:
        print("OpenAI api error:", e)
        return "ERROR: Cant get responce from ChatGPT"
