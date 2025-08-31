from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.6)

prompt1=PromptTemplate(
    template="generate a detailed report on {topic}",
    input_variables=["topic"]
)
prompt2=PromptTemplate(
    template="generate a 5 point summary on the following \n {text} \n in no more than 100 words.",
    input_variables=["text"]
)

parser = StrOutputParser()
chain = prompt1 | model | parser | prompt2 | model | parser
print(chain.invoke({"topic":"hockey"})) 
chain.get_graph().print_ascii()