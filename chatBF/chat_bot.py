import os
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.embeddings.base import Embeddings
from langchain.callbacks.base import BaseCallbackHandler
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# 自定义角色设定
system_prompt = """
你是BinFai交易助手，一名专业的加密货币交易问答专家。
你具备扎实的技术分析知识和金融知识，擅长解读K线、指标、策略、交易系统搭建等问题。
请用简洁、清晰、实用的语言回答用户问题。
如果无法确定答案，请坦诚说明而不是编造。
请不要输出思考过程，直接给出答案。
如果一定要展示思考过程，请用【思考过程开始】和【思考过程结束】包裹。
最终回答用【回答开始】标记开始。
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template("上下文：{context}\n用户问题：{question}")
])


def load_bge_base_zh_model():
    model_path = r"D:\zhaohuakun\TRANSFORMERS_CACHE\models--BAAI--bge-base-zh\snapshots\0e5f83d4895db7955e4cb9ed37ab73f7ded339b6"  # 你下载缓存路径
    model = SentenceTransformer(model_path)
    return model

class BgeBaseZhEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        # 输入文本列表，返回列表的向量列表
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        # 输入单条文本，返回向量
        return self.model.encode([text])[0].tolist()

class FilterThoughtsCallback(BaseCallbackHandler):
    def __init__(self):
        self.buffer = ""
        self.output_started = False

    def on_llm_new_token(self, token: str, **kwargs):
        # 假设模型输出中“思考”用特殊标记包裹，你需要根据实际内容调整判断逻辑
        # 这里只是示范，假设“【回答开始】”标记回答开始
        self.buffer += token
        if "【回答开始】" in self.buffer:
            self.output_started = True
            # 找到标记后，把标记后内容开始打印
            start_index = self.buffer.index("【回答开始】") + len("【回答开始】")
            print(self.buffer[start_index:], end="", flush=True)
            self.buffer = ""  # 清理缓存
        elif self.output_started:
            # 输出后续token
            print(token, end="", flush=True)

    def on_llm_end(self, response, **kwargs):
        self.buffer = ""
        self.output_started = False
        print()

def load_vector_store(persist_path: str, embedding_model):
    vector_store = FAISS.load_local(persist_path, embedding_model, allow_dangerous_deserialization=True)
    return vector_store

def build_qa_chain(vector_store, llm_model_name="qwen3:8b"):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model=llm_model_name,
                    streaming=True,
                    callbacks=[FilterThoughtsCallback()],
                    temperature=0.7)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return qa_chain

def chat_loop(qa_chain):
    print("欢迎使用BinFai交易问答助手")
    while True:
        query = input("\nuser：")
        if query.lower() == "exit":
            print("感谢使用")
            break
        qa_chain.invoke({"question": query})

if __name__ == "__main__":
    model = load_bge_base_zh_model()
    embedding = BgeBaseZhEmbeddings(model)
    # 这里写你的路径
    persist_folder = "../faiss"

    # 加载向量库
    vector_store = load_vector_store(persist_folder, embedding_model=embedding)

    # 构建多轮对话QA链
    qa_chain = build_qa_chain(vector_store)

    # 启动对话
    chat_loop(qa_chain)
