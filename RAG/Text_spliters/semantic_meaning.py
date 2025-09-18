from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# Initialize HuggingFace embeddings
# You can swap with any sentence-transformers model like "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create SemanticChunker
semantic_splitter = SemanticChunker(embeddings,
                                    breakpoint_threshold_type="standard_deviation",
                                    breakpoint_threshold_amount=0.95
                                    )

# Example text
text = """
Artificial Intelligence (AI) is transforming industries worldwide. 
In healthcare, AI assists doctors in diagnosing diseases more accurately 
by analyzing medical images and patient records.

In finance, algorithms can detect fraudulent transactions in real time, 
providing more security for customers. 
They also help investors make better decisions by analyzing large volumes of market data.

Education is another field being reshaped by AI. 
Intelligent tutoring systems can adapt lessons according to a student’s progress, 
ensuring personalized learning experiences.

However, AI also raises important ethical questions. 
Concerns about data privacy, job displacement, and algorithmic bias 
must be addressed to ensure responsible development.

Overall, AI holds great promise, 
but society must strike a balance between innovation and responsibility.
"""

# Split into semantic chunks
chunks = semantic_splitter.create_documents([text])

# Print chunks
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i} ---")
    print(chunk.page_content)
    print()
