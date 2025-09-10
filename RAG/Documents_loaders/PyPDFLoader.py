from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('R-Programming - Arithmetic.pdf')

doc = loader.load()

print(len(doc))
print(doc[22].page_content)
print(doc[22].metadata)
