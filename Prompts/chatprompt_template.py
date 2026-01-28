from langchain_core.prompts import ChatPromptTemplate



chat_template =ChatPromptTemplate(
    [  
        ('system', "You are a helpful {domain} expert."),
        ('human', "Explain in Simple terms what is {user_query}"),
    ]
)
prompt=chat_template.invoke({
    "domain": "machine learning",
    "user_query": "overfitting in machine learning"
})
print(prompt)