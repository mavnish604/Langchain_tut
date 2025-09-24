from langchain_community.tools import ShellTool

s=ShellTool()

print(s.invoke("whoami"))