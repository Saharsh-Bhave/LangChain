from langchain_openai import OpenAI # package where the interaction code b/w langchain and openAI is written
from dotenv import load_dotenv # package to manage keys

load_dotenv() # load the API key

llm = OpenAI(model='gpt-3.5-turbo-instruct') #openai object stored in a variable named llm

result = llm.invoke("How old is Eminem?") # invoke function used to give prompt to the model with the o/p being stored in a variable named result. 

print(result)