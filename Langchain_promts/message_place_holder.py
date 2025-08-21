from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
chat_template = ChatPromptTemplate.from_messages([
    ("system","you are helpful customer support agent"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human",'{query}')
])
chat_history=[]
with open("chat_history.txt") as f:
    chat_history.extend(f.readlines())
# print(chat_history)
prompt=chat_template.invoke({'chat_history':chat_history,'query':"where is my refund"})
print(prompt.to_string())