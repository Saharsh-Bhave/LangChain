from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Optional, Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

#schema

class Review(TypedDict):
    
    key_themes: Annotated[list[str], "Write down all the key themes mentioned in the review."]
    summary: Annotated[str, "A brief summary of the review."]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review, either positive, negative or neutral."]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list."] # optional tells this is optional as review might not have pros/cons.
    cons: Annotated[Optional[list[str]], "Write down all the cons in a list."]

with_structured_model = model.with_structured_output(Review)
 
result = with_structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultre, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast-whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera-the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:

Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging S-Pen support is unique and useful

Cons:

Bulky and heavy-not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors""")

#print(result)
#print(result['summary'])
print(result['sentiment'])