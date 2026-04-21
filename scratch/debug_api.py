import asyncio
import os
from openai import AsyncOpenAI
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

async def test_openai():
    print("Testing OpenAI...")
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
        print(f"OpenAI Success: {resp.choices[0].message.content}")
    except Exception as e:
        print(f"OpenAI Failed: {e}")

async def test_gemini():
    print("Testing Gemini...")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        resp = await model.generate_content_async("Hello")
        print(f"Gemini Success: {resp.text}")
    except Exception as e:
        print(f"Gemini Failed: {e}")

async def main():
    await test_openai()
    await test_gemini()

if __name__ == "__main__":
    asyncio.run(main())
