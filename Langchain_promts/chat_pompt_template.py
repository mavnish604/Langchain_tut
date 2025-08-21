from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage
chat_template = ChatPromptTemplate([
    ("system","you are a helpful {domain} expert"),
    ("human","Explain in simple terms what is {topic}")
    ])
promt=chat_template.invoke({'domain':"cricket","topic":"WTC points system"})
print(promt)