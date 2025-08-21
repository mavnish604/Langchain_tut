#!/home/tst_imperial/langchain/venv/bin/python
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

messages = [
    SystemMessage(content="you are a helpful assistant"),
    HumanMessage(content="Tell me about langchain")
]
result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)