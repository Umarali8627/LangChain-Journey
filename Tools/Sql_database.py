from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.output_parsers import StrOutputParser
db = SQLDatabase.from_uri(
    'mssql+pyodbc://@DESKTOP-C59CMS2/unirag?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'
    )
print(db.get_usable_table_names())
sql_tool= QuerySQLDataBaseTool(db=db)
# define the model 
llm = HuggingFaceEndpoint(
    repo_id='MiniMaxAI/MiniMax-M2.1',
    task='text-generation',
)
model= ChatHuggingFace(llm=llm)
parser = StrOutputParser()
agent= create_sql_agent(
    llm=model,
    db=db,
    verbose=True,
    handle_parsing_error=True,
    agent_type="tool-calling"
)
# response = agent.invoke("can you tell me umar ali attendence in database system ")
# print(response["output"])
def ask_query(question):
    responese = agent.invoke(question)
    return responese
print("Hi ! there how may i help you today ")
while True:
    user_input= input("You : ")

    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Goodbye!")
        break
    else:
        response = ask_query(user_input)
        print(f'AI : {response['output']}')