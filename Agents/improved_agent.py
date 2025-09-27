from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.tools import Tool,StructuredTool
from langchain_community.utilities import GoogleSerperAPIWrapper
import python_weather
import asyncio
import os
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor,create_react_agent

load_dotenv()
api_key = os.getenv("SERPER_API_KEY")

search=GoogleSerperAPIWrapper()

search_tool= Tool(
    func=search.run,
    name="Search tool",
    description="used to search the web"
)

class Weather_IN(BaseModel):
    city:str=Field(required=True,description="The city for which weather data is needed")

def weather(city: str) -> str:
    async def fetch(city):
        async with python_weather.Client() as client:
            w = await client.get(city)
            return w

    return asyncio.run(fetch(city))


weather_tool=StructuredTool.from_function(
    func=weather,
    description="A function to fetch weather data of a particular city",
    args_schema=Weather_IN
)

llm=HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-120b',
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

prompt=hub.pull("hwchase17/react")


agent=create_react_agent(
    llm=model,
    tools=[weather_tool,search_tool],
    prompt=prompt
)

agent_executor=AgentExecutor(
    agent=agent,
    tools=[weather_tool,search_tool],\
    verbose=True
)

print(agent_executor.invoke({"input":"what is the current temp in the capital of uttrakahand?"}))

