from openai import AzureOpenAI
import os
from dotenv import load_dotenv
 
def generate_embeddings(text):
  
  load_dotenv("secrets.env")
  
  client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
  )

  model_name = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_ADA")

  return client.embeddings.create(input = [text], model=model_name).data[0].embedding