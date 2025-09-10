from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

prompt = PromptTemplate(
    template="tell me something about this under 500 words\n{document}",
    input_variables=["document"]
)

parser=StrOutputParser()

loader = TextLoader("placement.txt",encoding="utf-8")

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({"document":docs[0].page_content}))
