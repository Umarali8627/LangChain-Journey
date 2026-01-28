from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
# embeddings for a document

embeddings =OpenAIEmbeddings(model="text-embedding-3-small",dimensions=32)
documents=[
    "Islanabad is the capital of Pakistan",
    "Londo is the capital of UK",
    "Riydah is the capital of Saudi Arabia"
]
# 
response =embeddings.embed_documents(documents)

print(response)