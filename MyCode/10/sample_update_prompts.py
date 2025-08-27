from llama_index.core import SummaryIndex, SimpleDirectoryReader
from llama_index.core import PromptTemplate
from llama_index.core import Settings
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
documents = SimpleDirectoryReader("files").load_data()
summary_index = SummaryIndex.from_documents(documents)
qe = summary_index.as_query_engine()
print(qe.query("Who burned Rome?"))
print("------------------------")

new_qa_template = (
    "Context information is below."
    "---------------------"
    "{context_str}"
    "---------------------"
    "Given the context information "
    "and any of your prior knowledge, "
    "answer the query."
    "Query: {query_str}"
    "Answer:")

template = PromptTemplate(new_qa_template)

qe.update_prompts(
    {"response_synthesizer:text_qa_template": template}
)
print(qe.query("Who burned Rome?"))
