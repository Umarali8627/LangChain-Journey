from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
# embeddings for a just a single query

embeddings =OpenAIEmbeddings(model="text-embedding-3-small",dimensions=32)

response =embeddings.embed_query("Islanabad is the capital of Pakistan")

print(response)