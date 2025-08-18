from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1",
                          task="text-generation",
                          temperature=0.5                  
    )
model = ChatHuggingFace(llm=llm)
st.header("Research Tool")
usr_in=st.text_input("enter your thoughts!")

if st.button("summarize"):
    res=model.invoke(usr_in)
    st.write(res.content)
