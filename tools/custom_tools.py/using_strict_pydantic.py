from langchain.tools import StructuredTool
from pydantic import BaseModel,Field

class FactorialTool(BaseModel):
    n : int = Field(required=True,description="Enter a no to calculate its factorial")


def fact(n:int)->int:
    if(n<=1):
        return 1
    return fact(n-1)*n;

def factorial(n:int)->int:
    return fact(n)

factorial_tool = StructuredTool.from_function(
    func=factorial,
    name="factorial",
    description="calls a helper fn to return the factorial of the given interger",
    args_schema=FactorialTool
)

res = factorial_tool.invoke({"n":5})

print(res)
print(factorial_tool.name)
print(factorial_tool.description)
print(factorial_tool.args)