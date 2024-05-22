import os
import openai
from azure.identity import DefaultAzureCredential, AzureDeveloperCliCredential
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

# Replace these with your own values, either in environment variables or directly here
AZURE_SEARCH_SERVICE = "search-sanderstrothmann"
AZURE_SEARCH_INDEX = "index-sanderstrothmann"
# AZURE_SEARCH_INDEX_1 = "vector-1715913242600"
AZURE_OPENAI_SERVICE = "cog-kguqugfu5p2ki"
AZURE_OPENAI_CHATGPT_DEPLOYMENT = "chat"
AZURE_SEARCH_API_KEY = "i7F5uuUzXR8KCZ58o4r3aZAr9QG5dDp3erOLgz6kb9AzSeAabEHy"
AZURE_OPENAI_EMB_DEPLOYMENT = "embedding"

AZURE_CLIENT_ID = "c4642a73-05e3-4a68-8228-7d241ba8d6e6"
AZURE_CLIENT_SECRET = "I_F8Q~MhKD9fCfT9725j9mCad39G6bpwVpolAb.f"
AZURE_TENANT_ID = "667439c9-20b5-4283-bd7b-fb6b3099d221"
AZURE_SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")

# # Used by the OpenAI SDK
openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
openai.api_version = "2023-09-01-preview"
# # Comment these two lines out if using keys, set your API key in the OPENAI_API_KEY environment variable and set openai.api_type = "azure" instead
openai.api_type = "azure"
openai.api_key = "4657af893faf48e5bd81208d9f87f271"

storage_connection_string = "DefaultEndpointsProtocol=https;AccountName=sasanderstrothmann;AccountKey=x4eeHxz6VMBqpmE+eLmA8ECKvA1EzTeUzOH2b9GkLiW7TVeo8DPrx1ckbcMM2QCj+u06a8vkxbI4+AStDI0lAQ==;EndpointSuffix=core.windows.net"
# container_name = "conversation"

# AZURE_OPENAI_CLIENT = openai.AzureOpenAI(
#         api_key = "4657af893faf48e5bd81208d9f87f271",  
#         api_version = "2023-05-15",
#         azure_endpoint =f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
#     )

AZURE_STORAGE_ACCOUNT = "sasanderstrothmann"
storagekey = "g2LVDKMxRjz09R2t/CQTz2JwAWZcsCge/dsMlOXb2mo2adikxqPNQpbDk8yeSlaoP7C+dvLxtEAV+AStYsDMWQ=="
formrecognizerservice = "pick-ai-doc-intel-version-2"
formrecognizerkey = "e739eef01ab34d46b16bb69e879a14b6"
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

container = "data-source"
MAX_SECTION_LENGTH = 1000
SENTENCE_SEARCH_LIMIT = 100
SECTION_OVERLAP = 100

azd_credential = AzureDeveloperCliCredential() if AZURE_TENANT_ID is None else AzureDeveloperCliCredential(tenant_id=AZURE_TENANT_ID, process_timeout=60)
default_creds = azd_credential if AZURE_SEARCH_API_KEY is None or storagekey is None else None
search_creds = default_creds if AZURE_SEARCH_API_KEY is None else AzureKeyCredential(AZURE_SEARCH_API_KEY)
use_vectors = novectors

storage_creds = default_creds if storagekey is None else storagekey

formrecognizer_creds = default_creds if formrecognizerkey is None else AzureKeyCredential(formrecognizerkey)


def blob_name_from_file_page(filename, page = 0):
    if len(re.findall(".pdf", str(filename))) > 0:
        return filename.name + f"-{page}" + ".pdf"
    else:
        return filename.name

def upload_blobs(filename):
    blob_service = BlobServiceClient(account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net", credential=storage_creds)
    blob_container = blob_service.get_container_client(container)
    if not blob_container.exists():
        blob_container.create_container()

    # if file is PDF split into pages and upload each page as a separate blob
    if len(re.findall(".pdf", str(filename))) > 0:
        reader = PdfReader(filename)
        pages = reader.pages
        for i in range(len(pages)):
            blob_name = blob_name_from_file_page(filename, i)
            print(f"\tUploading blob for page {i} -> {blob_name}")
            f = io.BytesIO()
            writer = PdfWriter()
            writer.add_page(pages[i])
            writer.write(f)
            f.seek(0)
            blob_container.upload_blob(blob_name, f, overwrite=True)
    else:
        blob_name = blob_name_from_file_page(filename)
        blob_container.upload_blob(blob_name, overwrite=True)

def table_to_html(table):
    table_html = "<table>"
    rows = [sorted([cell for cell in table.cells if cell.row_index == i], key=lambda cell: cell.column_index) for i in range(table.row_count)]
    for row_cells in rows:
        table_html += "<tr>"
        for cell in row_cells:
            tag = "th" if (cell.kind == "columnHeader" or cell.kind == "rowHeader") else "td"
            cell_spans = ""
            if cell.column_span > 1: cell_spans += f" colSpan={cell.column_span}"
            if cell.row_span > 1: cell_spans += f" rowSpan={cell.row_span}"
            table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
        table_html +="</tr>"
    table_html += "</table>"
    return table_html

def compute_embedding_text_3_large(text):
    response = openai.Embedding.create(engine=AZURE_OPENAI_EMB_DEPLOYMENT, input=text)["data"][0]["embedding"]
    return response

def get_document_text(filename):
    offset = 0
    page_map = []
    if localpdfparser:
        reader = PdfReader(filename)
        pages = reader.pages
        for page_num, p in enumerate(pages):
            page_text = p.extract_text()
            page_map.append((page_num, offset, page_text))
            offset += len(page_text)
    else:
        print(f"Extracting text from '{filename}' using Azure Form Recognizer")
        form_recognizer_client = DocumentAnalysisClient(endpoint=f"https://{formrecognizerservice}.cognitiveservices.azure.com/", credential=formrecognizer_creds, headers={"x-ms-useragent": "azure-search-chat-demo/1.0.0"})
        with open(filename, "rb") as f:
            poller = form_recognizer_client.begin_analyze_document("prebuilt-layout", document = f)
        form_recognizer_results = poller.result()

        for page_num, page in enumerate(form_recognizer_results.pages):
            tables_on_page = [table for table in form_recognizer_results.tables if table.bounding_regions[0].page_number == page_num + 1]

            # mark all positions of the table spans in the page
            page_offset = page.spans[0].offset
            page_length = page.spans[0].length
            table_chars = [-1]*page_length
            for table_id, table in enumerate(tables_on_page):
                for span in table.spans:
                    # replace all table spans with "table_id" in table_chars array
                    for i in range(span.length):
                        idx = span.offset - page_offset + i
                        if idx >=0 and idx < page_length:
                            table_chars[idx] = table_id

            # build page text by replacing charcters in table spans with table html
            page_text = ""
            added_tables = set()
            for idx, table_id in enumerate(table_chars):
                if table_id == -1:
                    page_text += form_recognizer_results.content[page_offset + idx]
                elif table_id not in added_tables:
                    page_text += table_to_html(tables_on_page[table_id])
                    added_tables.add(table_id)

            page_text += " "
            page_map.append((page_num, offset, page_text))
            offset += len(page_text)

    return page_map

def split_text(page_map):
    SENTENCE_ENDINGS = [".", "!", "?"]
    WORDS_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]
    print(f"Splitting '{filename}' into sections")

    def find_page(offset):
        num_pages = len(page_map)
        for i in range(num_pages - 1):
            if offset >= page_map[i][1] and offset < page_map[i + 1][1]:
                return i
        return num_pages - 1

    all_text = "".join(p[2] for p in page_map)
    length = len(all_text)
    start = 0
    end = length
    while start + SECTION_OVERLAP < length:
        last_word = -1
        end = start + MAX_SECTION_LENGTH

        if end > length:
            end = length
        else:
            # Try to find the end of the sentence
            while end < length and (end - start - MAX_SECTION_LENGTH) < SENTENCE_SEARCH_LIMIT and all_text[end] not in SENTENCE_ENDINGS:
                if all_text[end] in WORDS_BREAKS:
                    last_word = end
                end += 1
            if end < length and all_text[end] not in SENTENCE_ENDINGS and last_word > 0:
                end = last_word # Fall back to at least keeping a whole word
        if end < length:
            end += 1

        # Try to find the start of the sentence or at least a whole word boundary
        last_word = -1
        while start > 0 and start > end - MAX_SECTION_LENGTH - 2 * SENTENCE_SEARCH_LIMIT and all_text[start] not in SENTENCE_ENDINGS:
            if all_text[start] in WORDS_BREAKS:
                last_word = start
            start -= 1
        if all_text[start] not in SENTENCE_ENDINGS and last_word > 0:
            start = last_word
        if start > 0:
            start += 1

        section_text = all_text[start:end]
        yield (section_text, find_page(start))

        last_table_start = section_text.rfind("<table")
        if (last_table_start > 2 * SENTENCE_SEARCH_LIMIT and last_table_start > section_text.rfind("</table")):
            # If the section ends with an unclosed table, we need to start the next section with the table.
            # If table starts inside SENTENCE_SEARCH_LIMIT, we ignore it, as that will cause an infinite loop for tables longer than MAX_SECTION_LENGTH
            # If last table starts inside SECTION_OVERLAP, keep overlapping
            print(f"Section ends with unclosed table, starting next section with the table at page {find_page(start)} offset {start} table start {last_table_start}")
            start = min(end - SECTION_OVERLAP, start + last_table_start)
        else:
            start = end - SECTION_OVERLAP

    if start + SECTION_OVERLAP < end:
        yield (all_text[start:end], find_page(start))


def index_sections(filename, index_name):
    upload_blobs(filename)
    page_maps = get_document_text(filename)
    for page_map in page_maps:
        content = page_map[2]
        pagenum = len(page_map)
        section_data = [{   
            'id':str(uuid.uuid4()),
            'category': None,
            'sourcepage': blob_name_from_file_page(filename, pagenum),
            'sourcefile': filename.name,
            'content':content, 
            'embedding': compute_embedding_text_3_large(content)
        }]
        print(section_data)
        # section = json.dumps(section_data)
        # section_df = pd.DataFrame(section_data)
        # section = section_df.to_dict(orient='records')
        search_client = SearchClient(endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net/",
                                    index_name=index_name,
                                    credential=search_creds)
        search_client.upload_documents(documents=section_data)
        # sourcepage = blob_name_from_file_page(filename, pagenum)

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
