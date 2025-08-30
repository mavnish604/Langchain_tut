from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()

llm = HuggingFaceEndpoint(repo_id="NousResearch/Hermes-4-405B",
    task="text-generation")
model=ChatHuggingFace(llm=llm)
parser=JsonOutputParser()
template = PromptTemplate(
    template="Give me the name,age and city of a finctional person \n {format_ins}",
    input_variables=[],
    partial_variables={"format_ins":parser.get_format_instructions()}
)
chain = template | model | parser 
print(chain.invoke({}))                                                              