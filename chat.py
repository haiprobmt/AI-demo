import os
import openai
from azure.identity import AzureDeveloperCliCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from azure.core.credentials import AzureKeyCredential
import re
import json
import tiktoken
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
import uuid
from pypdf import PdfReader, PdfWriter
import html
import io
from azure.ai.formrecognizer import DocumentAnalysisClient
# from tenacity import retry, stop_after_attempt, wait_random_exponential
import tempfile
import streamlit as st

# Replace these with your own values, either in environment variables or directly here
AZURE_SEARCH_SERVICE = st.secrets.AZURE_SEARCH_SERVICE
AZURE_SEARCH_INDEX = st.secrets.AZURE_SEARCH_INDEX
# AZURE_SEARCH_INDEX_1 = "vector-1715913242600"
AZURE_OPENAI_SERVICE = st.secrets.AZURE_OPENAI_SERVICE
AZURE_OPENAI_CHATGPT_DEPLOYMENT = st.secrets.AZURE_OPENAI_CHATGPT_DEPLOYMENT
AZURE_SEARCH_API_KEY = st.secrets.AZURE_SEARCH_API_KEY
AZURE_OPENAI_EMB_DEPLOYMENT = st.secrets.AZURE_OPENAI_EMB_DEPLOYMENT

AZURE_CLIENT_ID = st.secrets.AZURE_CLIENT_ID
AZURE_CLIENT_SECRET = st.secrets.AZURE_CLIENT_SECRET
AZURE_TENANT_ID = st.secrets.AZURE_TENANT_ID
AZURE_SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")

# # Used by the OpenAI SDK
openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
openai.api_version = "2023-09-01-preview"
# # Comment these two lines out if using keys, set your API key in the OPENAI_API_KEY environment variable and set openai.api_type = "azure" instead
openai.api_type = "azure"
openai.api_key = st.secrets.api_key

storage_connection_string = st.secrets.storage_connection_string
# container_name = "conversation"

# AZURE_OPENAI_CLIENT = openai.AzureOpenAI(
#         api_key = "4657af893faf48e5bd81208d9f87f271",  
#         api_version = "2023-05-15",
#         azure_endpoint =f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
#     )

AZURE_STORAGE_ACCOUNT = st.secrets.AZURE_STORAGE_ACCOUNT
storagekey = st.secrets.storagekey
formrecognizerservice = st.secrets.formrecognizerservice
formrecognizerkey = st.secrets.formrecognizerkey
verbose = True
novectors = True
remove = True
removeall = False
skipblobs = False
localpdfparser = True
TIKTOKEN_ENCODING = tiktoken.encoding_for_model("gpt-35-turbo-16k-0613")

def search(prompt, filter=None):
    credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    # Set up clients for Cognitive Search and Storage
    search_client = SearchClient(
        endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
        index_name=AZURE_SEARCH_INDEX,
        credential=credential)   
    
    query_vector = openai.Embedding.create(engine=AZURE_OPENAI_EMB_DEPLOYMENT, input=prompt)["data"][0]["embedding"]
    # filter = f"image eq '{image}'"
    r = search_client.search(prompt, 
                            filter=filter,
                            query_type=QueryType.SIMPLE, 
                            query_language="en-us", 
                            query_speller="lexicon", 
                            semantic_configuration_name="default", 
                            top=10,
                            vector=query_vector if query_vector else None, 
                            top_k=50 if query_vector else None,
                            vector_fields="embedding" if query_vector else None
                            )
    results = [doc['image'] + ": " + doc['content'].replace("\n", "").replace("\r", "") for doc in r if doc['image'] != None]
    content = "\n".join(results)
    user_message = prompt + "\n SOURCES:\n" + content
    
    # Regular expression pattern to match URLs
    url_pattern = r'https?://[^\s,]+(?:\.png|\.jpg|\.jpeg|\.gif)'
    # Find all URLs in the text
    image_urls = re.findall(url_pattern, content)
    if len(image_urls) > 0:
        image = image_urls[0]
    else:
        image = None
    return {"user_message": user_message, "image": image}

def search_demo(prompt, filter=None):
    credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    # Set up clients for Cognitive Search and Storage
    search_client = SearchClient(
        endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
        index_name="index-demo",
        credential=credential)   
    
    query_vector = openai.Embedding.create(engine=AZURE_OPENAI_EMB_DEPLOYMENT, input=prompt)["data"][0]["embedding"]
    # filter = f"image eq '{image}'"
    r = search_client.search(prompt, 
                            filter=filter,
                            query_type=QueryType.SIMPLE, 
                            query_language="en-us", 
                            query_speller="lexicon", 
                            semantic_configuration_name="default", 
                            top=3,
                            vector=query_vector if query_vector else None, 
                            top_k=50 if query_vector else None,
                            vector_fields="embedding" if query_vector else None
                            )
    results = [doc['sourcepage'] + ": " + doc['content'].replace("\n", "").replace("\r", "") for doc in r if doc['sourcepage'] != None]
    content = "\n".join(results)
    user_message = prompt + "\n SOURCES:\n" + content
    return user_message

def send_message(messages, model=AZURE_OPENAI_CHATGPT_DEPLOYMENT):
    response = openai.ChatCompletion.create(
        engine=model,
        messages=messages,
        temperature=0.0,
        max_tokens=1024
    )
    response_final = response['choices'][0]['message']['content']
    return response_final

def upload_conversation_to_blob(blob_name, data):
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
    # Convert dict to JSON
    if '.json' in blob_name:
        json_data = json.dumps(data)
    else:
        json_data = data
    # Get blob client
    blob_client = blob_service_client.get_blob_client("conversation", blob_name)

    # Upload the JSON data
    blob_client.upload_blob(json_data, overwrite=True)

def load_conversation(blob_name):
    # Create a BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)

    # Get a reference to the container
    container_client = blob_service_client.get_container_client("conversation")

    # Get a reference to the blob
    blob_client = container_client.get_blob_client(blob_name)

    # Download the blob as a text string
    json_data = blob_client.download_blob().readall()

    # Convert the JSON string to a Python object
    json_object = json.loads(json_data)

    # Now you can work with the JSON object
    return json_object

def delete_conversation(blob_name):
    # Create a BlobServiceClient object using the connection string
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)

    # Get a reference to the container
    container_client = blob_service_client.get_container_client("conversation")

    # Get a reference to the blob
    blob_client = container_client.get_blob_client(blob_name)

    # Delete the blob
    blob_client.delete_blob()

def get_blob_url_with_sas(file_name, container):
    # Generate the SAS token for the file
    sas_token = generate_blob_sas(
        account_name="sasanderstrothmann",
        account_key="QtoEp5hl3aIWHdkTO1Q8I4R30M5lNnrKsSHjkuAL6BMKvf03Vh6BJfJ5RWEG7hlAGRRu3/pvK+Kx+AStgTMMQQ==",
        container_name=container,
        blob_name=file_name,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now() + timedelta(hours=1)  # Set the expiry time for the SAS token
    )

    # Construct the URL with SAS token
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
    # Get a reference to the container
    container_client = blob_service_client.get_container_client(container)
    blob_url = container_client.get_blob_client(file_name).url
    blob_url_with_sas = f"{blob_url}?{sas_token}"
    return blob_url_with_sas

def upload_to_blob_storage(file):
    # Define your Azure Blob Storage connection string
    connect_str = storage_connection_string

    # Create a BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Define your container name
    container_name = container

    # Create a ContainerClient object
    container_client = blob_service_client.get_container_client(container_name)

    # Create a temporary file to save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.getvalue())

    # Upload the file to Azure Blob Storage
    with open(temp_file.name, "rb") as data:
        blob_client = container_client.upload_blob(name=file.name, data=data, overwrite=True)

    # Delete the temporary file
    temp_file.close()

    # Return the URL of the uploaded file
    return blob_client.url