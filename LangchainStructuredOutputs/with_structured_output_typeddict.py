from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

#schema

class Review(TypedDict):

    summary: str
    sentiment: str

with_structured_model = model.with_structured_output(Review)
 
result = with_structured_model.invoke("""I recently purchased the EcoBrew Smart Coffee Maker, and it has completely changed my mornings. The setup was straightforward, and the app integration works flawlessly. I love being able to schedule my brew time or start it remotely from my phone. The coffee tastes fantastic, and the machine is surprisingly quiet. It’s also eco-friendly with reusable filters, which is a big plus. The only downside is that the water reservoir is a bit small for heavy coffee drinkers. Overall, this is a smart, stylish, and efficient addition to my kitchen. Highly recommended for tech-savvy coffee lovers.""")

print(result)
print(result['summary'])
print(result['sentiment'])