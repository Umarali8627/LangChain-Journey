from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# define the prompt1

prompt1 = PromptTemplate(
    template="Genrate a detailed report on {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Genrate a 5 pointer summary from the following text  {text}",
    input_variables=["text"]
)

# now define model
llm= HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-3B-Instruct',
        task='text-generation',
)

model =ChatHuggingFace(llm=llm)
# now create the parser 
parser= StrOutputParser()

# now create chain Pipeline
chain = prompt1  | model | parser | prompt2 | model | parser 

# now get the result 
result = chain.invoke({"topic":"unemployement in Pakistan"})

print(result)

# now see the structure of our chain 
chain.get_graph().print_ascii()
