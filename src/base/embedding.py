from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.base.ZhuPuAiEmbeddings import ZhuPuAiEmbedding


class SentenceEmbedding:
    def __init__(self, file_path: str):
        self.file_path = file_path
        docs = PyPDFLoader(file_path).load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        sentences = text_splitter.split_documents(docs)
        self.vectorstore = Chroma.from_documents(sentences, ZhuPuAiEmbedding())

    def get_vectorstore(self):
        return self.vectorstore

    def search(self, query: str) -> str:
        docs = self.vectorstore.similarity_search(query)
        return docs[0].page_content


if __name__ == '__main__':
    a = SentenceEmbedding("C:\\Users\\16922\\Desktop\\文档1.pdf")
    print(a.search("We will dedicate a segment of our project to discussing these ethical issues"))
