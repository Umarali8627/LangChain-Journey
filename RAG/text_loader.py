from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# define the model or llm
llm= HuggingFaceEndpoint(
     repo_id='meta-llama/Llama-3.2-3B-Instruct',
     task='text-generation',
)
model=ChatHuggingFace(llm=llm)

# create a prompt 
prompt = PromptTemplate(
    template="Write a summary for the followeing poem.\n {poem}",
    input_variables=["poem"]
)
# define a parser for structured output 
parser = StrOutputParser()


# first create the loader object 
laoder = TextLoader('RAG/cricket.txt',encoding='utf-8')


# now laod the documents through the load function 
docs =laoder.load()
# print(type(docs))
# print(len(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)
# now create a chain 
chain = prompt | model | parser 

print(chain.invoke({'poem':docs[0]}))
