# Analayzing feedback on review if sentiment is positive so we show thank u for this and if negative then show something else 
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain_classic.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
# define the model 
llm= HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-3B-Instruct',
        task='text-generation',
)
model = ChatHuggingFace(llm=llm)

parser= StrOutputParser()

# class Feedback(BaseModel):
#     sentiment:Literal['positive','negative'] =Field(description="Give the sentiment of the  feedback")
# parser_2 = PydanticOutputParser(pydantic_object=Feedback)

prompt_1= PromptTemplate(
    template="Classify the sentiment of the following feedback text only  into a single word either positive or negative \n {feedback} ",
    input_variables=["feedback"],
    # partial_variables={
    #     "format_instruction": parser_2.get_format_instructions()
    # }
)

classifier_chain = prompt_1 | model | parser

prompt_2 = PromptTemplate(
    template ="Write an appropriate response to this positive feedback in two or one line  \n {feedback} ",
    input_variables=["feedback"]
)
prompt_3 = PromptTemplate(
    template ="Write an appropriate response to this negative feedback  in two or one line \n {feedback} ",
    input_variables=["feedback"]
)
branch_chain = RunnableBranch(
    (lambda x:x=='Positive',prompt_2 |model |parser),
    (lambda x:x=='Negative',prompt_3 |model |parser),
    RunnableLambda(lambda x: "could not find sentiment")
)
merge_chain = classifier_chain | branch_chain

result = merge_chain.invoke({'feedback':'This is a  good  smart phone'})
print(result)