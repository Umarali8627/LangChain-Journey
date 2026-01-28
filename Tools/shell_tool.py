from langchain_community.tools import ShellTool
from langchain_community.tools import sql_database

shell_tool = ShellTool()

results = shell_tool.invoke('whomai')

print(results)