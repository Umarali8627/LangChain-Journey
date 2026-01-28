from langchain_huggingface import HuggingFaceEmbeddings


embeddings =HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# now create a list of documents
documents=[
    "Islanabad is the capital of Pakistan",
    "London is the capital of UK",
    "Riyadh is the capital of Saudi Arabia"

]
# get embeddings for the documents
response =embeddings.embed_documents(documents)
print(str(response))
