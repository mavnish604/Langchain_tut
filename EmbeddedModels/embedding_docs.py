from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
docs=["india is located in south asia","new delhi is capital of india","dehradun is the captital of uttrakhand"]
vectors = embeddings.embed_documents(docs)
print(str(vectors))