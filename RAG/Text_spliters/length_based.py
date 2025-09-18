from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/home/tst_imperial/langchain/RAG/Documents_loaders/Text_spliters/Questions.pdf")

docs = loader.load()



splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap=0,
    separator=" "
)

x=splitter.split_documents(docs)
print(x[23].page_content)


