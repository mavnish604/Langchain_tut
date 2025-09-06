from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


prompt1 = PromptTemplate(
    template="write a joke on the topic: {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(template="explain this joke {text}",input_variables=["text"])
parser = StrOutputParser()

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)

print(chain.invoke({"topic":"langchain"}))

