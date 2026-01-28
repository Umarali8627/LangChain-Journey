from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-oss-20b",
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.4,
    max_completion_tokens=20
)

response = llm.invoke("Suggest me 5 male pakistani names")
print(response.content)
