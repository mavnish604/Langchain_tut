#!/home/tst_imperial/langchain/venv/bin/python
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Always load from project root
load_dotenv(dotenv_path="/home/tst_imperial/langchain/.env")

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

model = ChatOpenAI(
    api_key=groq_key,
    base_url="https://api.groq.com/openai/v1",
    model="llama3-70b-8192",
    temperature=2,
    max_completion_tokens=1000
)
 
result=model.invoke("what kind of horror is 13b does it ahve ghosts?")
print(result.content)