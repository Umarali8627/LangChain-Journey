from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# now create the model first 

llm= HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-3B-Instruct',
    task='text-generation',
)

model= ChatHuggingFace(llm=llm)

# create the 1st prompt 
template= PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=['topic']
)
# create the 2nd prompt 
template2=PromptTemplate(
    template="write a 5 line summary on following text./n {text}",
    input_variables=['text']
)

# now create the parser
parser= StrOutputParser()
# now create a chain for  define flow 
chain= template | model | parser | template2 |model | parser 

result = chain.invoke({'topic':'black hole '})

print(result)