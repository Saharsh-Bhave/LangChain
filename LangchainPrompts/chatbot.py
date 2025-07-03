#ChatBot with no memory of the previous conversation.
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

conversations = [
    SystemMessage(content="You are a helpful assistant")

] # adding memory using messages

while True:
    user_input = input("You: ")
    conversations.append(HumanMessage(content = user_input))
    if user_input == "exit" :
        break
    result = model.invoke(conversations)
    conversations.append(AIMessage(content = result.content))
    print("ChatBot: ", result.content)

print(conversations)