from langchain_core.tools import tool

'''
Step 1
We need to define a function with docs string
although the doc string is not nessasry but it 
would be helful for llm to understand 
what our function exactly does.

Step 2
Add type hints to make easier for the llm to understand 
what type of data to enter and what to expect in output

step 3
Add tool decorator

'''

def fact(n : int) -> int:
    """
    This is a helping fn
    """
    if(n==1):
        return 1
    return fact(n-1)*n

@tool
def factorial(n:int)->int:
    """
    This function takes an arugment n which is a integer 
    and returns its factorial
    """
    return fact(n)

res = factorial.invoke({"n":5})
 
print(res)
print(factorial.name)
print(factorial.description)
print(factorial.args)