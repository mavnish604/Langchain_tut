from langchain_community.document_loaders import WebBaseLoader

from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

prompt = PromptTemplate(
    template="tell me {question} about this under 500 words\n{document}",
    input_variables=["document","questions"]
)

parser=StrOutputParser()


loader = WebBaseLoader("https://en.wikipedia.org/wiki/Cricket")

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({"question":"How many stumps are there in football","document":docs}))
