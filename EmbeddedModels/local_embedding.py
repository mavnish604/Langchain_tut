
from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
text=["who tf are you?","xt",'ls']
vec=embedding.embed_documents(text)
print(str(vec))