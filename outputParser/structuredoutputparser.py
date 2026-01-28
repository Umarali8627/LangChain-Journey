from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema 

# Define the model 
llm = HuggingFaceEndpoint(
     repo_id='meta-llama/Llama-3.2-3B-Instruct',
    task='text-generation',
)
model = ChatHuggingFace(llm=llm)

# creating the schema 
schema = [
    ResponseSchema(name='fact_1',description="fact 1 of the topic"),
    ResponseSchema(name='fact_1',description="fact 2 of the topic"),
    ResponseSchema(name='fact_1',description="fact 3 of the topic")
]
# now create the parser 
parser = StructuredOutputParser.from_response_schema(schema)

template = PromptTemplate(
    template="Give 3 facts about the topic \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={
        'format_instruction':parser.get_format_instruction()
    }
)

#  now create the prompt
prompt= template.invoke({"topic":"black hole "})

result= model.invoke(prompt)

f_result=parser.parse(result.content)
# now print the final result 
print(f_result)