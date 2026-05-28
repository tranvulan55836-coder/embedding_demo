import numpy as np

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query_vec, corpus_vecs, top_k=3):
    """
    语义搜索：返回 top_k 最相关文本索引
    """
    similarities = [cosine_similarity(query_vec, vec) for vec in corpus_vecs]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return top_indices, [similarities[i] for i in top_indices]