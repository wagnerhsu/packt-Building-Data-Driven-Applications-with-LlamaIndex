from llama_index.core import Document, SummaryIndex, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Configure Ollama embedding model
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434",
    request_timeout=120.0
)

# Configure Ollama LLM
Settings.llm = Ollama(
    model="llama3.2:1b",
    base_url="http://localhost:11434",
    request_timeout=120.0
)

# Create local document content about Messi instead of using Wikipedia
messi_content = """
Lionel Andrés Messi is an Argentine professional footballer who plays as a forward for Ligue 1 club Paris Saint-Germain and captains the Argentina national team. Widely regarded as one of the greatest players of all time, Messi has won a record seven Ballon d'Or awards, a record six European Golden Shoes, and in 2020 was named to the Ballon d'Or Dream Team.

Born and raised in central Argentina, Messi relocated to Spain to join Barcelona at age 13, for whom he made his competitive debut aged 17 in October 2004. He established himself as an integral player for the club within the next three years, and in his first uninterrupted season in 2008–09 he helped Barcelona achieve the first treble in Spanish football; that year, aged 22, Messi won his first Ballon d'Or.

Messi has spent his entire professional career with Barcelona, where he has won a club-record 35 trophies, including ten La Liga titles, seven Copa del Rey titles and four UEFA Champions League titles. A prolific goalscorer and creative playmaker, Messi holds the records for most goals in La Liga (474), the Supercopa de España (14), the UEFA Super Cup (3) and is the player with the most official recorded assists in football history (315).

With Argentina, Messi won the 2021 Copa América and the 2022 FIFA World Cup. Prior to winning the World Cup, Messi had reached five finals with Argentina but was on the losing side each time.
"""

documents = [Document(text=messi_content)]
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)
index = SummaryIndex(nodes)
query_engine = index.as_query_engine()
print("Ask me anything about Lionel Messi!")

while True:
    question = input("Your question: ")
    if question.lower() == "exit":
        break
    response = query_engine.query(question)
    print(response)
