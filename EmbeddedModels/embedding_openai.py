from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
vector = embeddings.embed_query("Hello world")
print(vector)  
