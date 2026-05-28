from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str, model: str = "text-embedding-3-small"):
    """
    调用 OpenAI API 获取文本 embedding
    """
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding