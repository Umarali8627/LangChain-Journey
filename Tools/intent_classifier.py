from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# define the model 
llm =lm = HuggingFaceEndpoint(
    repo_id='MiniMaxAI/MiniMax-M2.1',
    task='text-generation',
)
model= ChatHuggingFace(llm=llm)
# design a prompt 
template = """Classify the intent of the user query {query} into one of:
- student_academic_data
- instructor_academic_data
- university_information
- events_and_timetable
- general_chat

Return only the intent."""
# now use prompt template 
prompt= PromptTemplate(
    template=template,
    input_variables=["query"]
)
parser = StrOutputParser()

chain = prompt | model | parser

def get_intent(query):
     
     response =chain.invoke({"query":query})
     return response

while True :
     user_input = input("Enter your Question ")

     if user_input.lower() in ["quit", "q"] :
          print('bye')
          break
     
     result= get_intent(user_input)
     if result == 'student_academic_data':
            print("Calling Sql Agent ")
     elif result == 'instructor_academic_data':
           print("Calling Sql Agent")
     elif result == 'university_information':
           print("RAG System Activating")
     elif result == 'events_and_timetable':
           print("RAG System Activating")
     elif result == 'general_chat':
           print("Response from LLM activate")
     else:
           print("Sorry, I couldn't understand your request.")
     
     



