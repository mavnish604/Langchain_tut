from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
from langchain_core.prompts import PromptTemplate
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="NousResearch/Hermes-4-405B",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact_1",description="fact 1 about the topic"),
    ResponseSchema(name="fact_2",description="fact 2 about the topic"),
    ResponseSchema(name="fact_3",description="fact 3 about the topic")
]
parser=StructuredOutputParser.from_response_schemas(schema)
template1=PromptTemplate(
    template="give me a detailed description on the \n {topic}",
    input_variables=["topic"]
)
template2=PromptTemplate(
    template="from this {text} give me top 3 most useful facts \n {format_ins}",
    input_variables=["text"],
    partial_variables={"format_ins":parser.get_format_instructions()}
)
chain = template1 | model |template2 | model | parser
print(chain.invoke({"topic":"aryan invasion theory is a hoax"}))
