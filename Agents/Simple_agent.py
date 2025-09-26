from dotenv import load_dotenv
from langchain_core.tools import Tool
import requests
import os
from langchain import hub
from langchain.agents import create_react_agent,AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv()

import os

api_key = os.getenv("SERPER_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

x=GoogleSerperAPIWrapper()

search_tool= Tool(
    name="Google search tool",
    func=x.run,
    description="search the web using this tool"
)

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=model,
    tools=[search_tool],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True
)

response = agent_executor.invoke({"input":"what is the most effiecent way to go from makka wala dehradun to dhanlauti"})

print(response)