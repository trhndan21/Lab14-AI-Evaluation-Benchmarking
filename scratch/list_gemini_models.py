import asyncio
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

async def list_models():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
    except Exception as e:
        print(f"Failed to list models: {e}")

if __name__ == "__main__":
    asyncio.run(list_models())
