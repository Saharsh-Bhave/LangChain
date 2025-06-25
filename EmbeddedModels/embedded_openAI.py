from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-small', dimensions= '32') # dimesions to tell the no. of dimensions in the vector output

result = embedding.embed_query("J.Cole is the best rapper.")

print(str(result))