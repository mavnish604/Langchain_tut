#!/home/tst_imperial/langchain/venv/bin/python
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt
load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.4)
st.header("Research Tool")
usr_in_paper=st.selectbox("Select Research Paper Name",["Attention is all you need","BERT: Pre-training of Deep bidirectional Transformers","GPT-3: Language Models are Few-shot Learning","Diffusion Models GANs on Image Synthesis"])
usr_in_style=st.selectbox("Select a Expalination style",["Begineer","Technical","code-oriented","Mathematical"])
usr_in_length=st.selectbox("Select Explaination length",["Short(about 2 paras)","Medium(about 5 paras)","long and detailed"])
template=load_prompt('/home/tst_imperial/langchain/template.json')
if st.button("summarize"):
    chain = template|model
    res=chain.invoke({
        'usr_in_paper':usr_in_paper,
        'usr_in_style':usr_in_style,
        'usr_in_length':usr_in_length
    })
    st.write(res.content)
