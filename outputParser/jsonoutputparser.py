from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser 


# create the model 
llm= HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-3B-Instruct',
    task='text-generation',
)
model =ChatHuggingFace(llm=llm)

# define the parser
parser = JsonOutputParser()

# now create the prompt 
template = PromptTemplate(
    template="Give me  5 facts about {topic}\n{format_instruction}",
    input_variables=[],
    partial_variables={
        "format_instruction":parser.get_format_instructions()
    }
)
# prompt = template.format()

# result =model.invoke(prompt)

# final_result =parser.parse(result.content)

# creat Chain instead of above code 
chain = template | model | parser 
result =chain.invoke({'topic':'black Hole'})
# print(final_result)
# print(type(final_result))
print(result)
# flaws of json 
# they dont have enforce to follow the schema 
