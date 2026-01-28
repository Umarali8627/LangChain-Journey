from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage



# first define the model 
def load_model():
    llm= HuggingFaceEndpoint(
        repo_id='meta-llama/Llama-3.2-3B-Instruct',
        task='text-generation',
    )
    return ChatHuggingFace(llm=llm)
# create the model 
model = load_model()
chat_history=[
    SystemMessage(content="You are a helpful AI assistant."),

]

while True:
    user_input= input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the chat. Goodbye!")
        break
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI :", result.content)


print(chat_history)

# how many types of messages in langchain 
# 3 types system_messages, human_messages, ai_messages