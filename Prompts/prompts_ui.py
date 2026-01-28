import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline,HuggingFaceEndpoint

st.set_page_config(page_title="Local LLM Chat", layout="centered")
st.header("Chat Model using Local HuggingFace Model")

@st.cache_resource
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

user_input = st.text_input("Enter your message here:")

if st.button("Send") and user_input:
    response = model.invoke(user_input)
    st.write(response.content)
