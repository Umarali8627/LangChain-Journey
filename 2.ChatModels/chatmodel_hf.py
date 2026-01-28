from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# deepseek-ai/DeepSeek-V3.2
# Qwen/Qwen3-4B-Instruct-2507
llm= HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-3B-Instruct',
    task='text-generation',
    # max_new_tokens=20,
)

model= ChatHuggingFace(llm=llm)

result=model.invoke("Does light need medium to travel?")

print(result.content)




