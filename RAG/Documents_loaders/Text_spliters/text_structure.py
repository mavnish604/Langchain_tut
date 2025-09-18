from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader


docs = PyPDFLoader("RAG/Documents_loaders/Text_spliters/Questions.pdf").load()

chunks = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0
).split_documents(docs)

print(chunks[3].page_content)