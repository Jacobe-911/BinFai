from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import os
import numpy as np

def recursive_load_md_files(root_dir):
    documents = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".md"):
                file_path = os.path.join(dirpath, fname)
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    print(f"加载文件失败: {file_path}, 错误: {e}")
    return documents

class SentenceTransformerEmbeddings:
    def __init__(self, model_path_or_name):
        print(f"加载SentenceTransformer模型: {model_path_or_name}")
        self.model = SentenceTransformer(model_path_or_name)

    def embed_documents(self, texts):
        # texts是list[str]，返回 list[list[float]]
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embeddings.tolist()

def build_vector_store(docs_path: str, persist_path: str, embedding_model_path: str):
    documents = recursive_load_md_files(docs_path)
    print(f"加载文档数量: {len(documents)}")
    if len(documents) == 0:
        print("没有加载到任何文档，请检查路径和文件编码！")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print(f"切分成 {len(docs)} 个文本块")

    embeddings = SentenceTransformerEmbeddings(embedding_model_path)

    # 生成所有文本块向量
    texts = [doc.page_content for doc in docs]
    vectors = embeddings.embed_documents(texts)
    print(f"生成了 {len(vectors)} 个向量，每个向量维度 {len(vectors[0])}")

    # 这里用FAISS直接从向量构建索引
    # FAISS.from_documents 其实是内部帮你先做embedding再建库
    # 但现在我们已有向量，需要用from_texts + embeddings构建或直接用from_embeddings构建
    # 下面示例用from_texts，因为我们已经实现了embed_documents，会自动调用
    vector_store = FAISS.from_texts(texts, embeddings)

    os.makedirs(persist_path, exist_ok=True)
    vector_store.save_local(persist_path)
    print(f"向量库已保存到 {persist_path}")

if __name__ == "__main__":
    os.environ["TRANSFORMERS_CACHE"] = r"D:\zhaohuakun\TRANSFORMERS_CACHE"  # 确保设置了缓存目录
    docs_folder = "../knowledge_base"
    persist_folder = "../faiss"
    embedding_model_name = "BAAI/bge-base-zh"  # 你下载并缓存好的模型名或路径

    build_vector_store(docs_folder, persist_folder, embedding_model_name)
