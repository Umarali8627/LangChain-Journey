from langchain_community.document_loaders import PyPDFLoader

# create the loader object 
loader =PyPDFLoader('RAG/Deep_Learning_Notes.pdf')

# now laod the file 
docs= loader.load()
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)