import os

from langchain_core.callbacks import StreamingStdOutCallbackHandler, CallbackManager
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.runnables import RunnablePassthrough

from src.init.property import Property
from src.base.embedding import SentenceEmbedding

# 初始化prompt工程
_prompt = {
    'prompt_1': ChatPromptTemplate.from_template(
        """你是问答任务的助手。使用以下检索到的上下文来回答问题。如果你不知道答案，就说你不知道。最多使用三个句子并保持答案简洁。

        {context}

        Question: {question}

        Helpful Answer:"""
    ),
    'prompt_2': hub.pull("rlm/rag-prompt")
}


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class RAG:
    # 初始化ZHIPU API KEY
    os.environ["ZHIPUAI_API_KEY"] = Property.get_property("API_KEY")

    # 初始化模型
    _llm = ChatZhipuAI(
        temperature=0.95,
        model="glm-4"
    )

    __streaming_chat = ChatZhipuAI(
        model="glm-4",
        temperature=0.5,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

    def __init__(self, file_path: str):
        try:
            global _prompt
            self._example_prompt = _prompt["prompt_1"]
        except NameError:
            pass
        self.__file_path = file_path
        self.__sentenceEmbedding = SentenceEmbedding(file_path)
        self._retriever = self.__sentenceEmbedding.get_vectorstore().as_retriever(search_type="similarity",
                                                                                  search_kwargs={"k": 5})

        try:
            self.__rag_chain = (
                    {"context": self._retriever | format_docs, "question": RunnablePassthrough()}
                    | self._example_prompt
                    | RAG._llm
                    | StrOutputParser()
            )
        except AttributeError:
            pass

    def get_chat(self, message: str):
        for chunk in self.__rag_chain.stream(message):
            print(chunk, end="", flush=True)

    def select_prompt(self, prompt_index: int = 1):
        global _prompt
        prompt_name = "prompt_" + str(prompt_index)
        self._example_prompt = _prompt[prompt_name]


if __name__ == '__main__':
    r = RAG("C:\\Users\\16922\\Desktop\\文档1.pdf")
    r.select_prompt(2)
    r.get_chat("what can Multimodal Agent AI systems do?")
