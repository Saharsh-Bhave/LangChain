#Dynamic prompt
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

model = ChatOpenAI(model='gpt-4')

st.header("Research Tool")

paper_input = st.selectbox("Select research paper name", ["Select...", "Attention is all you need", "BERT: Pre-Training of Deep Biderictional Transformers", "GPT-3: Language Models are a Few Short Learners", "Diffusion Models Beat GANs on Image Synthesis"])
style_input = st.selectbox("Select your style", ["Select...", "Beginner-Friendly", "Technical", "Code-oriented", "Mathematical"])
length_input = st.selectbox("Select your length", ["Select...", "Short(1-2 paragraphs)", "Medium(3-4 paragraphs)", "Long(Detailed explanation)"])

template = load_prompt('template.json')

#fill the placeholder
prompt = template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})

if st.button("Summarizer"):
    result = model.invoke(prompt)
    st.write(result.content)