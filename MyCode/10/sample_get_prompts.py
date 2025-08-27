from llama_index.core import SummaryIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI

Settings.llm = OpenAI(
    api_key="fake-key",
    api_base="http://localhost:1234/v1"
)
documents = SimpleDirectoryReader("files").load_data()
summary_index = SummaryIndex.from_documents(documents)
qe = summary_index.as_query_engine()

prompts = qe.get_prompts()

for k, p in prompts.items():
    print(f"Prompt Key: {k}")
    print("Text:")
    print(p.get_template())
    print("\n")
