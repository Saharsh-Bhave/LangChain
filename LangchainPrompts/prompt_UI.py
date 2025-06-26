#static prompt
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatOpenAI(model='gpt-4')

st.header("Research Tool")

user_input = st.text_input("Enter your query.")

if st.button("Summarizer"):
    result = model.invoke(user_input)
    st.write(result.content)