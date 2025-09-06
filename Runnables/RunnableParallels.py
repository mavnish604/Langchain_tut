from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

from langchain.schema.runnable import RunnableParallel,RunnableSequence

load_dotenv()

model_g = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    max_new_tokens=200
)

model_l = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="write a x tweet on the topic {topic} under word limit",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="generate a highly detailed linkedin post on the topic {topic}",
    input_variables=["topic"]
)

parser=StrOutputParser()

chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1,model_g,parser),
    "linkedin":RunnableSequence(prompt2,model_l,parser)
})

print(chain.invoke({"topic":"Agentic AI"})["tweet"])
print(chain.invoke({"topic":"Agentic AI"})["linkedin"])