from langchain_openai import ChatOpenAI  # Chat model of open AI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=0.7, max_completion_tokens=25) # temperature governs how creative/deterministic the answer will be, and max_completion_tokens governs the number of tokens in the answer.

result = model.invoke("Name the 5 best basketball players of all time.")

print(result.content) #adding ".content" gives only the answer and not all the meta info.