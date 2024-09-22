import pickle
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever

def initialize_retrievers(embedder, faiss_path, bm25_path, weights=[0.2, 0.8]):
    db_embed = FAISS.load_local(faiss_path, embedder, allow_dangerous_deserialization=True)
    retriever = db_embed.as_retriever()

    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25, retriever], weights=weights)
    return ensemble_retriever
