#  we are going to make 3 steps Chain 
 # 1-> get prompt from user 
 # 2-> send to llm 
 # 3-> Display llm response 

# first import libraries 
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



# create prompttemplate 
prompt = PromptTemplate(
    template ="Genrate 5 intersting facts about {topic}",
    input_variables=["topic"]
)
# define the model 
llm= HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-3B-Instruct',
        task='text-generation',
)

model =ChatHuggingFace(llm=llm)
# parse the ouptut 
parser= StrOutputParser()

# now create the chain 
chain= prompt | model| parser
# get response from pipeline 
response = chain.invoke({"topic":"cricket"})
# print response 
print(response)
# now check the steps of chain 
chain.get_graph().print_ascii()
# this is a sequential chain 
