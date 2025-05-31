from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.embeddings.base import Embeddings
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

system_prompt = """
你是BinFai交易助手，一名专业的加密货币交易问答专家。
你具备扎实的技术分析知识和金融知识，擅长解读K线、指标、策略、交易系统搭建等问题。
请用简洁、清晰、实用的语言回答用户问题。
请**不要重复用户的问题**，直接给出回答内容。
如果无法确定答案，请坦诚说明而不是编造。
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template("上下文：{context}\n用户问题{question}")
])


def load_bge_base_zh_model():
    model_path = r"D:\zhaohuakun\TRANSFORMERS_CACHE\models--BAAI--bge-base-zh\snapshots\0e5f83d4895db7955e4cb9ed37ab73f7ded339b6"
    model = SentenceTransformer(model_path)
    return model


class BgeBaseZhEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


def load_vector_store(persist_path: str, embedding_model):
    vector_store = FAISS.load_local(persist_path, embedding_model, allow_dangerous_deserialization=True)
    return vector_store


def build_qa_chain(vector_store, llm_model_name="qwen2.5:7b"):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model=llm_model_name,
                    streaming=True,  # 关闭流式输出，改为直接返回结果
                    temperature=0.7)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return qa_chain


# === 全局初始化一次，避免每次调用重复加载 ===
model = load_bge_base_zh_model()
embedding = BgeBaseZhEmbeddings(model)
persist_folder = "../faiss"
vector_store = load_vector_store(persist_folder, embedding_model=embedding)
qa_chain = build_qa_chain(vector_store)


def chat_bot(question: str) -> str:
    # 使用 qa_chain.run 同步调用，返回结果字符串
    # 注意：run方法直接返回回答字符串
    answer = qa_chain.run(question)
    return answer


if __name__ == "__main__":
    print("欢迎使用BinFai交易问答助手，输入 exit 退出")
    while True:
        query = input("\nuser：")
        if query.lower() == "exit":
            print("感谢使用")
            break
        answer = chat_bot(query)
        print("BinFaiChatBot：", answer)
