from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

text = """Python Based:

1.Difference between mutable and immutable
objects in Python?
2.What will be the output of the following and why?
a = 256
b = 256
print(a is b)
x = 257
y = 257
print(x is y)
3. Predict the output:
def add(x, y=[]):
y.append(x)
return y
print(add(1))
print(add(2))
print(add(3, []))
print(add(4))
4.Without using a temporary variable, how do you
swap two numbers in Python?
5. What is the difference between these two
statements?
list1 = [1, 2, 3]list2 = list1
list3 = list1[:]
6. What will be the output?
print(bool("False"))
print(bool(""))
7. Why does this code behave differently?
print(0.1 + 0.2 == 0.3)
8. Predict the output and explain:
for i in range(3):
print(i)
else:
print("Finished")
9. What will be the output?
x = [1, 2, 3]
print(id(x))
x += [4, 5]
print(id(x))
y = (1, 2, 3)
print(id(y))
y += (4, 5)
print(id(y))
10. Write a one-liner in Python to reverse a string without using loops.ANSWERS:
1. Difference between mutable and immutable objects
in Python?
• Mutable → Can be changed after creation. Examples: list, dict, set.
• Immutable → Cannot be changed after creation. Examples: int, float, str, tuple.
lst = [1, 2, 3]
lst[0] = 10
# ✅ Works (mutable)
s = "hello"
# s[0] = "H" ❌ Error (immutable)
s = "Hello"
# Creates a new string
2. Output & Why?
a = 256
b = 256
print(a is b)
# True
x = 257
y = 257
print(x is y)
# False (in most implementations)
? Explanation:
• Python caches small integers in the range [-5, 256] for efficiency.
• So a and b point to the same memory.
• For numbers outside this range (like 257), Python creates new objects, so x is y → False.
3. Predict the output
def add(x, y=[]):
y.append(x)
return y
print(add(1))
# [1]print(add(2))# [1, 2](same default list reused!)
print(add(3, []))# [3](new list passed explicitly)
print(add(4))# [1, 2, 4]
? Explanation: Default mutable arguments (y=[]) are shared across function calls unless you
explicitly pass a new list.
4. Swap without temp variable
a, b = 10, 20
a, b = b, a
print(a, b)
# 20 10
? Python allows tuple unpacking for swapping.
5. Difference between the statements
list1 = [1, 2, 3]
list2 = list1# Reference copy → both point to same object
list3 = list1[:]# Shallow copy → new object with same elements
• list2 and list1 are the same object → modifying one affects the other.
• list3 is a different object (copy).
6. Output
print(bool("False"))# True (non-empty string is True)
print(bool(""))# False (empty string is False)
7. Why different behavior?
print(0.1 + 0.2 == 0.3)
# False
? Due to floating-point precision error.
Internally 0.1 + 0.2 = 0.30000000000000004, which is not exactly equal to 0.3.8. Predict the output
for i in range(3):
print(i)
else:
print("Finished")
Output:
0
1
2
Finished
? Explanation: The else in a for loop executes if the loop completes normally (not broken by
break).
9. Output
x = [1, 2, 3]
print(id(x))
# same id
x += [4, 5]
print(id(x))
# same id (list modified in-place)
y = (1, 2, 3)
print(id(y))
# some id
y += (4, 5)
print(id(y))
# different id (new tuple created)
? Explanation:
• Lists are mutable, so += modifies in-place → id stays same.
• Tuples are immutable, so += creates a new tuple object → id changes.
10. One-liner to reverse a string
s = "Python"
print(s[::-1])
# nohtyP
? Uses slicing with negative step."""



chunks = RecursiveCharacterTextSplitter.from_language(
    chunk_size = 150,
    chunk_overlap=0,
    language=Language.PYTHON
).split_text(text)

print(chunks[3])