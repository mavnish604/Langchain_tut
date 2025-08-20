from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-2.0-flash")
chat_history=[]
while(True):
    usr_in = input("you :").strip().lower()
    chat_history.append(usr_in)
    if usr_in =='exit':
        break
    res=model.invoke(chat_history)
    chat_history.append(res.content)
    print("AI: ",res.content)

print(chat_history)