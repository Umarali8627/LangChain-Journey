# chatmodel_hf_local.py
# now import the correct classes from langchain_huggingface,and use HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline


# now first define the llm with model id 
llm= HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        
    )
)
model= ChatHuggingFace(llm=llm) 
# now invoke the model 
response=model.invoke("Define Priority Sheduling in Operating Systems.")
# print the response content
print(response.content)
