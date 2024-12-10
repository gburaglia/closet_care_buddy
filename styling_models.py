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
from langchain_openai import OpenAI as LGOpenAI
from openai import OpenAI

import pandas as pd
import tempfile
import shutil
temp_dirs = []

def generatePicturePrompt(img_description):
    llm = LGOpenAI(temperature=0.9)

    # Sample description given by previous llm chain
    description = img_description

    system_prompt = (
            "Generate a detailed prompt to generate an image based on the following description: "
            "You are an experienced stylist. "
            "Make sure the image is full body and has all the components of an outfit (i.e. pants, shirt, shoes, accessories). "
            "Make sure it is for a casual context. "
            "\n\n"
            "{image_desc}"
        )

    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

    chain = prompt | llm  | StrOutputParser()

    out_message = chain.invoke({
            "image_desc" : description,
            "input": "Provide a full body outfit inspiration"
        })

    return out_message


def generatePicture(generated_prompt):
  client = OpenAI()

  response = client.images.generate(
    model="dall-e-3",
    prompt=generated_prompt,
    size="1024x1024",
    quality="standard",
    n=1,
  )

  image_url = response.data[0].url
  return image_url

def runStylingModels(img_description):
    generated_prompt = generatePicturePrompt(img_description)
    generated_img_url = generatePicture(generated_prompt)
    return generated_prompt, generated_img_url
