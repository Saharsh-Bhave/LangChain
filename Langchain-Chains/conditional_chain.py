from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback.')

parser2= PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)
prompt2 = PromptTemplate(
    template='Give an appropriate response for this positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template='Give an appropriate response for this negative feedback \n {feedback}',
    input_variables=['feedback']
)

classifier_chain = prompt1 | model | parser2

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find out the sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback': 'This is a womderful product'})
print(result)