from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch,RunnableParallel,RunnableSequence,RunnablePassthrough,RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()


model1 =  ChatGoogleGenerativeAI(model="gemini-2.5-flash")

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

model2 = ChatOpenAI(
    api_key=groq_key,
    base_url="https://api.groq.com/openai/v1",
    model="openai/gpt-oss-120b",
    max_completion_tokens=1000
)

prompt1 = PromptTemplate(
    template="write a detaliled report on the {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Summarise this text /n {text}",
    input_variables=["text"]
)

parser=StrOutputParser()

report_chain = RunnableSequence(prompt1,model1,parser)

branch=RunnableBranch(
    (lambda x:len(x.split())>100,RunnableSequence(prompt2,model1,parser)),
    RunnablePassthrough()
)

chain=RunnableSequence(report_chain,branch)

print(chain.invoke({"topic":"sandpaper gate of 2017"}))

