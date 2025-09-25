from Tool_binding import Factorial
from Tool_binding import model
from langchain_core.messages import HumanMessage

query = HumanMessage("can you calculate the factorial of 10 using factorial tool")

messages = [query]

messages

res = model.invoke(messages)
messages.append(res)
res1 = res.tool_calls[0]
res2 = res1["args"]
# print(Factorial.invoke(res1))
# #tool message
messages.append(Factorial.invoke(res1))
print(model.invoke(messages).content)

# #tool messages can be sent back to llm