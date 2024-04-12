import os
from dotenv import load_dotenv

OPENAI_API_KEY = [ENTER YOUR OPENAI API KEY HERE]
PINECONE_API_KEY = [ENTER YOUR PINECONE API KEY HERE]
PINECONE_API_ENV = [ENTER YOUR PINECONE API ENVIRONMENT HERE]

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#Loading the model
from langchain_openai.chat_models import ChatOpenAI

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

#Set up the string parser
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

chain = model | parser

#Create a template
from langchain.prompts import ChatPromptTemplate

template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

#Load the text into the model
from langchain_community.document_loaders import TextLoader

loader = TextLoader("transcription.txt")
text_documents = loader.load()

#Split the text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(text_documents)

#Embeddings 
from langchain_openai.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

#Computing Similarity 
from sklearn.metrics.pairwise import cosine_similarity

#Pinecone Confusion