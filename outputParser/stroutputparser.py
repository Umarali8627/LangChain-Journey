from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

llm= HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-3B-Instruct',
    task='text-generation',
)

model= ChatHuggingFace(llm=llm)
#1st prompt -> detailed reporet
template= PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=['topic']
)

#2nd  prompt -> 
template2=PromptTemplate(
    template="write a 5 line summary on following text./n {text}",
    input_variables=['text']
)


prompt1= template.invoke({"topic":"black hole"})

result1=model.invoke(prompt1)

prompt2 =template2.invoke({"text":result1.content})

result2=model.invoke(prompt2)

print(result2.content)

