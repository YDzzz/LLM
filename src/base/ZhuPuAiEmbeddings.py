import os
from abc import ABC
from typing import List

from zhipuai import ZhipuAI

from src.init.property import Property

from langchain_core.embeddings import Embeddings

os.environ["ZHIPUAI_API_KEY"] = Property.get_property("API_KEY")
client = ZhipuAI()


def _text_qualify(embedding_text):
    """
    using ZhipuAI Embedding API to get embedding 1024 dimension support
    :param embedding_text:
    :return:
    """
    if type(embedding_text) == str:
        e_t = embedding_text
    else:
        e_t = embedding_text.page_content
    # print usage of token number:
    #   response.usage.total_tokens
    # embedding support:
    #   response.data[0].embedding
    response = client.embeddings.create(
        model="embedding-2",
        input=e_t,
    )
    return response.data[0].embedding


class ZhuPuAiEmbedding(Embeddings, ABC):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i in texts:
            embeddings.append(_text_qualify(i))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return _text_qualify(text)

    def __init__(self):
        super().__init__()
