from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful research assistant.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []

#load chat history
with open('C:/Users/HP/OneDrive/Desktop/LangChainModels/LangchainPrompts/chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

#create prompt

prompt = chat_template.invoke({'chat_history':chat_history, 'query':"Where is my refund?"})

print(prompt) #this prompt can be used for a model