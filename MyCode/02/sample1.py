from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Configure Ollama embedding model
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434"
)

# Configure Ollama LLM with a very small model
Settings.llm = Ollama(
    model="llama3.2:1b",  # Very small 1B parameter model
    base_url="http://localhost:11434"
)
documents = SimpleDirectoryReader('files').load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("summarize each document in a few sentences")

print(response)
