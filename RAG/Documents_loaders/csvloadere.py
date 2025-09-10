from langchain_community.document_loaders import CSVLoader

loader=CSVLoader("path")

docs = loader.load()
