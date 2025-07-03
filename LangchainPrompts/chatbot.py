#ChatBot with no memory of the previous conversation.
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

conversations = []

while True:
    user_input = input("You: ")
    conversations.append(user_input)
    if user_input == "exit" :
        break
    result = model.invoke(conversations)
    conversations.append(result.content)
    print("ChatBot: ", result.content)

print(conversations)