#Dynamic prompt
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatOpenAI(model='gpt-4')

st.header("Research Tool")

paper_input = st.selectbox("Select research paper name", ["Select...", "Attention is all you need", "BERT: Pre-Training of Deep Biderictional Transformers", "GPT-3: Language Models are a Few Short Learners", "Diffusion Models Beat GANs on Image Synthesis"])
style_input = st.selectbox("Select your style", ["Select...", "Beginner-Friendly", "Intermediate", "Expert"])
length_input = st.selectbox("Select your length", ["Select...", "Short", "Medium", "Long"])



user_input = st.text_input("Enter your query.")

if st.button("Summarizer"):
    result = model.invoke(user_input)
    st.write(result.content)