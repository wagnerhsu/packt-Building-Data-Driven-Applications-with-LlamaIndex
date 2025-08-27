import os
from llama_index.llms.openai_like import OpenAILike
from dotenv import load_dotenv


# 推荐：使用 .env 文件管理 base_url 和 api_key
load_dotenv()
base_url = os.environ.get("BASE_URL", "http://localhost:1234/v1")
api_key = os.environ.get("API_KEY", "lm-studio")

# 将 model 替换为你在 LM Studio 正在服务的模型名
llm = OpenAILike(
    model="openai/gpt-oss-20b",
    api_key=api_key,
    api_base=base_url,
    temperature=0.3,  # 可调
    max_tokens=512,  # 可调
    max_retries=2,  # 网络/服务短暂失败时自动重试
)

resp = llm.complete("请问你是什么模型？")
print(resp.text)
