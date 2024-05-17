from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.serve.RAG import RAG

__contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
_contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", __contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

__qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
_qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", __qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


class HistoryRAG(RAG):

    def __init__(self, file_path):
        global _contextualize_q_prompt
        global _qa_prompt
        super().__init__(file_path)
        self.__chat_history = []

        history_aware_retriever = create_history_aware_retriever(
            HistoryRAG._llm, self._retriever, _contextualize_q_prompt
        )
        question_answer_chain = create_stuff_documents_chain(HistoryRAG._llm, _qa_prompt)
        self.__rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_chat(self, question: str):
        ai_msg = self.__rag_chain.invoke({"input": question, "chat_history": self.__chat_history})
        self.__chat_history.extend([HumanMessage(content=question), ai_msg["answer"]])
        return ai_msg["answer"]

    def clear_history(self):
        self.__chat_history.clear()

    def select_prompt(self, prompt_index: int = 1):
        pass


if __name__ == '__main__':
    hr = HistoryRAG("C:\\Users\\16922\\Desktop\\文档1.pdf")
    print(hr.get_chat("what can Multimodal Agent AI systems do?"))
    print(hr.get_chat(input()))

