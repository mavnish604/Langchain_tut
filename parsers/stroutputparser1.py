from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm = HuggingFaceEndpoint(repo_id="NousResearch/Hermes-4-405B",
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
parser = StrOutputParser()
chain = templte1 | model | parser | templte2 | model | parser
res=chain.invoke({"topic":"dark web"})
print(res)



