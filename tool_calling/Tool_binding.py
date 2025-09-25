from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from pydantic import BaseModel,Field
import requests

load_dotenv()


class FactorialInput(BaseModel):
    n:int=Field(required=True,description="A int to calculate its factorial.")

def fact(n:int)->int:
    if(n<=1):
        return 1
    return fact(n-1)*n

def factorial(n:int)->int:
    return fact(n)

Factorial = StructuredTool.from_function(
    func=factorial,
    name="factorial tool",
    description="This tool takes an integer input and return factorial of that number",
    args_schema=FactorialInput
)

# binding the tool

model = ChatHuggingFace(llm = HuggingFaceEndpoint(
    repo_id="NousResearch/Hermes-4-405B",
    task="text-generation"
))

model=model.bind_tools([Factorial])


