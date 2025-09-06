from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableLambda,RunnableSequence,RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

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

model1 = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def cnt_words(x):
    return len(x.split())


runnable_wrd_cnt = RunnableLambda(cnt_words)


prompt1=PromptTemplate(
    template="Write a joke on {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="explain this {joke}",
    input_variables=["joke"]
)

parser=StrOutputParser()


joke_gen = RunnableSequence(prompt1,model1,parser)

chain_para=RunnableParallel({
    "joke":RunnablePassthrough(),
    "length of joke":RunnableLambda(cnt_words),
    "explaination":RunnableSequence(prompt2,model2,parser)
}
)

final_chain = RunnableSequence(joke_gen,chain_para)

print(final_chain.invoke({"topic":"coffee"})["joke"])
print("-----------------------------------------------")
print(final_chain.invoke({"topic":"coffee"})["length of joke"])

print(final_chain.invoke({"topic":"coffee"})["explaination"])