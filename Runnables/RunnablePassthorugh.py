from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence,RunnablePassthrough,RunnableParallel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

model2 = ChatOpenAI(
    api_key=groq_key,
    base_url="https://api.groq.com/openai/v1",
    model="openai/gpt-oss-20b",
    max_completion_tokens=1000
)

model1 = ChatGoogleGenerativeAI(model="gemini-2.0-flash",max_tokens=100)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

prompt2= PromptTemplate(
    template="expalin this joke {joke}",
    input_variables=["joke"]
)

joke_chain = RunnableSequence(prompt1,model1,parser)

para_chain=RunnableParallel({
    "joke":RunnablePassthrough(),
    "explaination":RunnableSequence(prompt2,model2,parser)
})
final_chain = RunnableSequence(joke_chain,para_chain)
print(final_chain.invoke({"topic":"ChatGPT"})["joke"])
print(final_chain.invoke({"topic":"ChatGPT"})["explaination"])