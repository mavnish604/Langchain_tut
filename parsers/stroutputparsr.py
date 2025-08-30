from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b",
    task="text-generation")
model=ChatHuggingFace(llm=llm)

templte1= PromptTemplate(
    template="can you do a deepresearch on {topic}",
    input_variables=["topic"]
)
templte2=PromptTemplate(
    template="can you summarise {text} in 5 lines",
    input_variables=["text"]
)
promt1= templte1.invoke({"topic":"dark web"})
res1=model.invoke(promt1)
promt2=templte2.invoke({"text":res1.content})
res2=model.invoke(promt2)
print(res2.content)



