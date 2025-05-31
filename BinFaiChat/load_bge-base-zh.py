import os

os.environ["HF_HOME"] = r"D:\zhaohuakun\TRANSFORMERS_CACHE"

from sentence_transformers import SentenceTransformer

def load_bge_base_zh():
    model_name = "BAAI/bge-base-zh"
    print(f"正在加载模型 {model_name} ...")
    model = SentenceTransformer(model_name)
    print("模型加载完成。")
    return model

def get_embedding(model, text):
    embedding = model.encode(text)
    return embedding

if __name__ == "__main__":
    model = load_bge_base_zh()
    sample_text = "凉兮（Liangxi）是中国加密货币圈内知名的投资人、交易者和社区影响力人物，同时也是活跃的区块链自媒体创作者。他以深度的市场分析和实战经验著称，受到广大币圈投资者和交易者的关注。"
    emb = get_embedding(model, sample_text)
    print(f"文本: {sample_text}")
    print(f"向量维度: {len(emb)}")
    print(f"向量示例: {emb[:5]}")  # 打印向量前5个数值
