def calculate_precision_at_k(ranked_docs, relevant_docs, k):
    """
    Calculates the precision at k for the search results.
    Precision@k is the fraction of relevant documents in the top k ranked documents.
    """
    if k > len(ranked_docs):
        k = len(ranked_docs)
    top_k_docs = ranked_docs[:k]
    relevant_count = sum(1 for doc in top_k_docs if doc in relevant_docs)
    precision_at_k = relevant_count / k
    return precision_at_k

def calculate_r_precision(ranked_docs, relevant_docs):
    """
    Calculates the R-Precision for the search results.
    R-Precision is the Precision@k where k equals the number of relevant documents.
    """
    r = len(relevant_docs)
    retrieved_docs = ranked_docs[:r]
    relevant_retrieved = sum(1 for doc in retrieved_docs if doc in relevant_docs)
    r_precision = relevant_retrieved / r if r > 0 else 0
    return r_precision

