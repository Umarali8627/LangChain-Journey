import streamlit as st 
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel



st.set_page_config(page_title="Chat Bot ", layout="centered")
st.header("Quiz Notes Genrator")
# define the model
llm= HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-3B-Instruct',
        task='text-generation',
)
model_1= ChatHuggingFace(llm=llm)
model_2= ChatHuggingFace(llm=llm)
# now create a prompt 1 
prompt_1= PromptTemplate(
    template ="Genrate the short and simple notes for this topic \n {text}",
    input_variables=["text"]
) 
# now design the prompt 2 
prompt_2 = PromptTemplate(
    template="Genrate  5 short question Quiz and answer with simple and brief description from the following {text}",
    input_variables=["text"]
)
# now design prompt 3 
prompt_3 = PromptTemplate(
     template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz ->{quiz}",
    input_variables=["notes","quiz"]
)
# now parser
parser=StrOutputParser()
# now chain for parallel state 
parallel_chain = RunnableParallel({
    "notes": prompt_1 | model_1 | parser,
    "quiz":prompt_2 | model_2 | parser
})
# now merging 
merge_chain = prompt_3 | model_1 | parser

chain =parallel_chain | merge_chain
text =st.text_input("Entered the Topic to Create notes and Quiz")

if st.button("send"):
    st.write('genrating...')
    result=chain.invoke({"text":text})
    st.write(result)
