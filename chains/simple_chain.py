from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.6)
prompt = PromptTemplate(
    template="Generate 5 interseting facts about {topic}",
    input_variables=["topic"]
)
parser = StrOutputParser()

chain = prompt | model | parser
print(chain.invoke({"topic":"test carrier of rohit sharma"}))
chain.get_graph().print_ascii()