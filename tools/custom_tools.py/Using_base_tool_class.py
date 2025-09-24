from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel,Field

class FactorialToolInput(BaseModel):
    n : int = Field(required=True,description="Enter a no to calculate its factorial")

class FactorialTool(BaseTool):
    name:str = "FACTORIAL"
    description:str="RETURNS THE FACTORIAL OF TWO NUMBERS"
    args_schema:Type[BaseModel]=FactorialToolInput

    def fact(self,n:int)->int:
        if n<=1:
            return 1
        return self.fact(n-1)*n
    def _run(self,n:int)->int:
        return self.fact(n)
    
fact=FactorialTool()
res = fact.invoke({"n":7})
print(res)
print(fact.name)
print(fact.description)
print(fact.args)    