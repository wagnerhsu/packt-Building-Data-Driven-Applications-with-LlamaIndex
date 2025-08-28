from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.llms.openai_like import OpenAILike

print("Configuring LM Studio...")

Settings.llm = OpenAILike(
    api_key="not-needed",
    api_base="http://localhost:1234/v1",
    timeout=120.0,  # Increased timeout to 2 minutes
    max_retries=1,  # Reduced retries to avoid long waits
    model="openai/gpt-oss-20b",  # Specify model name
    temperature=0.7,
    max_tokens=512,
    request_timeout=60.0  # Additional timeout parameter
)

reader = SimpleDirectoryReader('files')
documents = reader.load_data()
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)

title_extractor = TitleExtractor(summaries=["self"]) 
meta= title_extractor.extract(nodes)

print("\nFirst title: " +meta[0]['document_title'])
print("Second title: " +meta[1]['document_title'])

