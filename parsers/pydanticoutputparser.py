from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

class person(BaseModel):
    name : str = Field(description="name of the person")
    age : int = Field(gt=18,description="age of the person")
    city : str = Field(description="name of the city where the person lives")

parser = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(
    template="generate the name of a fantasy {place} character along with his city and age it \n {format_ins}",
    input_variables=["place"],
    partial_variables={"format_ins":parser.get_format_instructions()}
)
chain=template|model|parser
r=chain.invoke({"place":"india"})
print(r)