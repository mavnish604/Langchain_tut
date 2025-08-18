#!/home/tst_imperial/langchain/venv/bin/python
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
# Always load from project root
load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
result=model.invoke("who am i")
print(result.content)