from llama_index.core import Document, SummaryIndex, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import time

# Configure Ollama embedding model with timeout
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434",
    request_timeout=120.0
)

# Configure Ollama LLM with timeout
Settings.llm = Ollama(
    model="llama3.2:1b",
    base_url="http://localhost:11434",
    request_timeout=120.0
)

try:
    loader = WikipediaReader()
    print("Loading Wikipedia data about Lionel Messi...")
    documents = loader.load_data(pages=["Lionel Messi"])
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(documents)
    index = SummaryIndex(nodes)
    query_engine = index.as_query_engine()
    print("Ready! Ask me anything about Lionel Messi!")
    
    while True:
        question = input("Your question: ")
        if question.lower() == "exit":
            break
        try:
            response = query_engine.query(question)
            print(response)
        except Exception as e:
            print(f"Error answering question: {e}")
            print("Please try again or type 'exit' to quit.")

except Exception as e:
    print(f"Error initializing the system: {e}")
    print("Please make sure Ollama is running and try again.")
