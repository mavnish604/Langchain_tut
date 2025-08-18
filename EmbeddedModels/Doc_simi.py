from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
emb = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
ml_docs = [
    "Supervised learning uses labeled datasets to train models to predict outcomes.",
    "Unsupervised learning finds hidden patterns or groupings in unlabeled data.",
    "Reinforcement learning trains agents to make decisions by rewarding good actions.",
    "Overfitting happens when a model learns training data too well but fails on unseen data.",
    "A confusion matrix summarizes classification results by comparing predictions with actual labels.",
    "Gradient descent is an optimization algorithm that minimizes the cost function in machine learning models.",
    "A decision tree splits data into branches to make predictions based on feature values.",
    "Neural networks are computational models inspired by the human brain, used for complex tasks like vision and NLP.",
    "Cross-validation is a technique for evaluating model performance by splitting data into multiple train-test sets.",
    "Principal Component Analysis (PCA) reduces the dimensionality of datasets while retaining most variance."
]
query="tell me about neural nets"
doc_emb=emb.embed_documents(ml_docs)
q_emb=emb.embed_query(query)
scores=cosine_similarity([q_emb],doc_emb)[0]
index,scores=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print("query:",query)
print(ml_docs[index])
print("similarity score is: ",scores)