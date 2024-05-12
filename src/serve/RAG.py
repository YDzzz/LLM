import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.callbacks import StreamingStdOutCallbackHandler, CallbackManager

from src.init.property import Property
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class RAG:
    # 初始化ZHIPU API KEY
    os.environ["ZHIPUAI_API_KEY"] = Property.get_property("API_KEY")

    # 初始化prompt工程
    __prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a world class technical documentation writer."),
        ("user", "{input}")
    ])

    # 初始化模型
    __llm = ChatZhipuAI(
        temperature=0.95,
        model="glm-4"
    )

    __streaming_chat = ChatZhipuAI(
        model="glm-4",
        temperature=0.5,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

    # 构建langchain
    chain = __prompt | __llm

    def __init__(self, file_path: str):
        self.__file_path = file_path
        loader = PyPDFLoader(self.__file_path)
        file = loader.load()

    def get_answer(self, message) -> str:
        return RAG.__llm.invoke(message).content

    def get_streaming_chat(self, message) -> str:
        return RAG.__streaming_chat.invoke(message).content


if __name__ == '__main__':
    r = RAG("C:\\Users\\16922\\Desktop\\文档1.pdf")
    print(r.get_streaming_chat("hello"))
    print(r.get_streaming_chat("what can you do"))
