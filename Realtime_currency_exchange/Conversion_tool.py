from pydantic import BaseModel,Field
import requests
from langchain.tools import StructuredTool
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from typing import Annotated
from langchain_core.tools import InjectedToolArg
load_dotenv()

import os

api_key = os.getenv("EXCHANGE_API")

class API_IN(BaseModel):
    base_curr:str=Field(required=True,description="The base currency to convert")
    target_curr:str=Field(required=True,description="The currency to convert")

def Get_rate(base_curr: str, target_curr: str) -> float:
    api_key = os.getenv("EXCHANGE_API")  # fetch from .env
    if not api_key:
        raise ValueError("API key not found. Make sure EXCHANGE_API is set in .env")

    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_curr}/{target_curr}"

    response = requests.get(url)
    data = response.json()

    if response.status_code != 200 or "conversion_rate" not in data:
        raise ValueError(f"Error fetching rate: {data}")

    return data

class Converter_Input(BaseModel):
    base_currency_val:float=Field(required=True,description="The value of the base currency to convert from")
    conversion_rate:float=Field(required=True,description="The conversion rate")

def convert(base_currency_val:float,conversion_rate:Annotated[float,InjectedToolArg])->float:
    return base_currency_val*conversion_rate

RATE = StructuredTool.from_function(
    func=Get_rate,
    name="GET_RATE",
    description="This function gets the conversion rate of the pair passed",
    args_schema=API_IN
)

CONVERSION = StructuredTool.from_function(
    func=convert,
    name="CONVERTOR",
    description="converts the the rate of one currency into the other it takes input of base currency and conversion rate",
    args_schema=Converter_Input
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools([CONVERSION,RATE])

query = HumanMessage("how much are 15.78 afgani in indian currency")

context = [query]

in_api = model.invoke(context)

rate = RATE.invoke(in_api.tool_calls[0])

context.append(in_api)

context.append(rate)

in_con=model.invoke(context)

context.append(in_con)
print(context)
context.append(CONVERSION.invoke(in_con.tool_calls[0]))
print(context)
parser = StrOutputParser()
res=model.invoke(context)
result = parser.invoke(res)
print(result)

