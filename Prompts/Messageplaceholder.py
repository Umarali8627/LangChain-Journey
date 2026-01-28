from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder


# chat prompt template
chat_template=ChatPromptTemplate.from_messages([
    ('system','You are a helpful customer support agent.'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human','{customer_query}'),
])
chat_history= []
# load the chat history
with open('Prompts/chat_history.txt','r') as f:
   chat_history.append(f.readlines())
print(chat_history)

# prompt=chat_template.invoke({
#    "chat_history": chat_history,
#    "customer_query": "where is my refund?"
# })
# print(prompt)