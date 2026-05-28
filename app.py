from embedding import get_embedding
from utils import semantic_search

# 示例文本
corpus = [
    "人工智能正在改变世界。",
    "机器学习是 AI 的一个重要分支。",
    "Python 是一门流行的编程语言。",
    "深度学习擅长处理图像和语音。",
    "量化交易依赖数据和算法。"
]

# 生成 corpus embedding
corpus_embeddings = [get_embedding(text) for text in corpus]

print("文本 embedding 已生成！\n")

while True:
    query = input("请输入查询 (exit 退出): ")
    if query.lower() == "exit":
        break

    query_vec = get_embedding(query)

    top_indices, scores = semantic_search(query_vec, corpus_embeddings, top_k=3)

    print("\n最相关文本:")
    for idx, score in zip(top_indices, scores):
        print(f"- {corpus[idx]} (相似度: {score:.4f})")
    print("\n")