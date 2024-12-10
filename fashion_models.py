from langchain_openai import ChatOpenAI
from transformers import pipeline
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings #for converting chunks into embeddings
from langchain_chroma import Chroma #database for stroring the embeddings
import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import ImageCaptionLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import re
import pandas as pd
import tempfile
import shutil
temp_dirs = []
fashion_caption_global = ""


def img2doc(img_path):
    
    # Create an ImageCaptionLoader instance
    loader = ImageCaptionLoader(img_path)
    # Load the caption as a document
    documents = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_dirs.append(temp_dir)  # Track temp directory for later cleanup
    
    # Create a new Chroma vectorstore
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings(), 
        persist_directory=temp_dir
    )
    retriever = vectorstore.as_retriever(k=1)
    
    return retriever, temp_dir


def cleanup_temp_dir(temp_dir):
    shutil.rmtree(temp_dir)


def textGeneration_langChain_RAG(img_path):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    system_prompt = (
        "You are an experienced clothing sylist. "
        "Use the following pieces of retrieved context to answer. "
        "Use two sentence maximum and be as detailed as possible yet concise. "
        "Include the clothing syle (i.e. bohemian, casual, classic, sporty, preppy). "
        "Include how new or worn the item looks to be. "
        "Be confident avoid using the word 'likely'. "
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    caption_retriever,temp_dir = img2doc(img_path)
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(caption_retriever, question_answer_chain)

    response = rag_chain.invoke({"input": "Describe the piece of clothing in this picture", "context": caption_retriever})

    cleanup_temp_dir(temp_dir)
    
    return response["answer"]

def runFashionModels(img_path):
    img_path_complete = "static/imgs/shots/" + img_path
    fashion_caption = textGeneration_langChain_RAG(img_path_complete)
    setFashionCaption(fashion_caption)
    return fashion_caption

# Function to set the fashion caption
def setFashionCaption(caption):
    global fashion_caption_global
    fashion_caption_global = caption

# Getter function for fashion_caption
def getFashionCaption():
    return fashion_caption_global
