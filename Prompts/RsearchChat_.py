from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import streamlit as st 
from langchain_core.prompts import PromptTemplate,load_prompt

def load_model():
    # llm = HuggingFacePipeline.from_model_id(
    #     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #     task="text-generation",
    #     pipeline_kwargs={
    #         "temperature": 0.5,
    #         "max_new_tokens": 512
    #     }
    # )
    llm= HuggingFaceEndpoint(
        repo_id='meta-llama/Llama-3.2-3B-Instruct',
        task='text-generation',
        # max_new_tokens=20,
    )
    return ChatHuggingFace(llm=llm)

model = load_model()
st.set_page_config(page_title="Research Paper Chatbot", layout="centered")
st.header("Research Tool ")

# first get the paper input from the user
paper_input= st.selectbox("Select the research paper:", 
                          options=["Attention is all you need", "Paper 2: Machine Learning Applications", "Paper 3: Natural Language Processing"])
# now get the style of summary
summary_style= st.selectbox("Select the summary style:", 
                            options=["Beginner Friendly", "Technical code oriented", "Bullet Points Summary"])
# now get the length of summary
summary_length= st.selectbox("Select the summary length:", 
                             options=["Short (1-2 paragraphs)", "Medium (300 words)", "Long (500 words)"])

# now create a prompt template
template = load_prompt("template.json")
# now fill the place holders



if st.button("Generate Summary"):
    # create a chain for the template and model 
    chain = template | model 
    # now get the response using the chain 
    response = chain.invoke({
    "paper_input": paper_input,
    "summary_style": summary_style,
    "summary_length": summary_length
})
    
    # display the response 
    st.write(response.content)
# prompt template benfits 
# 1. Reusability
# 2.Data Validation
# 3. Langchain Ecosystem Integration
