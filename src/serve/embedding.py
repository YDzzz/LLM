from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from text2vec import SentenceModel
from src.base.chinese_text_splitter import ChineseTextSplitter


class SentenceEmbedding:
    __model = SentenceModel('shibing624/text2vec-base-chinese')
    def __init__(self, file_path: str):
        self.file_path = file_path
        content = ""
        loader = PyPDFLoader(file_path)
        for page in loader.load():
            content += page.page_content
        sentences = ChineseTextSplitter(True).split_text(content)
        embeddings = SentenceEmbedding.__model.encode(sentences)
        self.vectorstore = Chroma.add_texts(iter(sentences), embeddings)

    def get_vectorstore(self):
        return self.vectorstore

    def search(self, query:str):
        embeddings = SentenceEmbedding.__model.encode(query)
        self.vectorstore


if __name__ == '__main__':
    SentenceEmbedding("C:\\Users\\16922\\Desktop\\文档1.pdf").
