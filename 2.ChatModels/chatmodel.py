from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='google/gemma-3-12b-it:free',
                               api_key="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",
                               base_url='https://openrouter.ai/api/v1',
                               )

result =model.invoke('what is the capital of Pakistan')
print(result.content)


