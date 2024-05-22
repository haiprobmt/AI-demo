# import base64
import html
import io
import re
import time
import streamlit as st
import openai
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureDeveloperCliCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient, ContentSettings
from pypdf import PdfReader, PdfWriter
from tenacity import retry, stop_after_attempt, wait_random_exponential
from io import BytesIO
import os

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

# AZURE_OPENAI_CLIENT = openai.AzureOpenAI(
#         api_key = "4657af893faf48e5bd81208d9f87f271",  
#         api_version = "2023-05-15",
#         azure_endpoint =f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
#     )

container = "data-source"
MAX_SECTION_LENGTH = 1000
SENTENCE_SEARCH_LIMIT = 100
SECTION_OVERLAP = 100

azd_credential = AzureDeveloperCliCredential() if AZURE_TENANT_ID is None else AzureDeveloperCliCredential(tenant_id=AZURE_TENANT_ID, process_timeout=60)
default_creds = azd_credential if AZURE_SEARCH_API_KEY is None or storagekey is None else None
search_creds = default_creds if AZURE_SEARCH_API_KEY is None else AzureKeyCredential(AZURE_SEARCH_API_KEY)
use_vectors = novectors

storage_creds = default_creds if storagekey is None else storagekey

if not localpdfparser:
    # check if Azure Form Recognizer credentials are provided
    if formrecognizerservice is None:
        print("Error: Azure Form Recognizer service is not provided. Please provide formrecognizerservice or use --localpdfparser for local pypdf parser.")
        exit(1)
    formrecognizer_creds = default_creds if formrecognizerkey is None else AzureKeyCredential(formrecognizerkey)

def blob_name_from_file_page(filename, page = 0):
    filename_name = filename.name
    if len(re.findall(".pdf", str(filename))) > 0:
            return filename_name + f"-{page}" + ".pdf"
    else:
        return filename_name

def upload_blobs(filename):
    blob_service = BlobServiceClient(account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net", credential=storage_creds)
    cnt_settings = ContentSettings(content_type="application/pdf", content_disposition= "inline")
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
            blob_container.upload_blob(blob_name, f, overwrite=True, content_settings=cnt_settings)
            get_blob_path = blob_container.get_blob_client(blob_name).url
            print(get_blob_path)
        
def blob_exists(container_name, blob_name):
    try:
        # Create a ContainerClient to interact with the container
        container_client = BlobServiceClient.get_container_client(container_name)

        # Check if the blob exists by attempting to get its properties
        blob_client = container_client.get_blob_client(blob_name)
        blob_properties = blob_client.get_blob_properties()
        return True

    except Exception as e:
        # If an exception is raised, the blob doesn't exist
        return False
def remove_blobs(filename):
    if verbose: print(f"Removing blobs for '{filename.name or '<all>'}'")
    blob_service = BlobServiceClient(account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net", credential=storage_creds)
    blob_container = blob_service.get_container_client(container)
    if blob_container.exists():
        if filename is None:
            blobs = blob_container.list_blob_names()
            for b in blobs:
                if verbose: print(f"\tRemoving blob {b}")
                blob_container.delete_blob(b)
        else:
            reader = PdfReader(filename)
            pages = reader.pages
            for i in range(len(pages)):
                blobs = blob_name_from_file_page(filename, i)
                if blob_exists(container, blobs):
                    if verbose: print(f"\tRemoving blob {blobs}")
                    blob_container.delete_blob(blobs)
                else:
                    if verbose: print(f"\tThere is no blob {blobs}")

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
        if verbose: print(f"Extracting text from '{filename.name}' using Azure Form Recognizer")
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
    # if verbose: print(f"Splitting '{filename}' into sections")

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
            if verbose: print(f"Section ends with unclosed table, starting next section with the table at page {find_page(start)} offset {start} table start {last_table_start}")
            start = min(end - SECTION_OVERLAP, start + last_table_start)
        else:
            start = end - SECTION_OVERLAP

    if start + SECTION_OVERLAP < end:
        yield (all_text[start:end], find_page(start))

def filename_to_id(filename):
    filename_ascii = re.sub("[^0-9a-zA-Z_-]", "_", filename.name)
    filename_hash = base64.b16encode(filename.name.encode('utf-8')).decode('ascii')
    return f"file-{filename_ascii}-{filename_hash}"

def create_sections(filename, page_map, use_vectors):
    file_id = filename_to_id(filename)
    for i, (content, pagenum) in enumerate(split_text(page_map)):
        section_data = {   
            'id': f"{file_id}-page-{i}",
            'category': None,
            'sourcepage': blob_name_from_file_page(filename, pagenum),
            'sourcefile': filename.name,
            'content':content, 
            'embedding': compute_embedding(content)
        }
        yield section_data

def before_retry_sleep(retry_state):
    if verbose: print("Rate limited on the OpenAI embeddings API, sleeping before retrying...")

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15), before_sleep=before_retry_sleep)
def compute_embedding(text):
    response = openai.Embedding.create(engine=AZURE_OPENAI_EMB_DEPLOYMENT, input=text)["data"][0]["embedding"]
    return response

def index_sections(filename, sections):
    if verbose: print(f"Indexing sections from '{filename.name}' into search index '{AZURE_SEARCH_INDEX}'")
    search_client = SearchClient(endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net/",
                                    index_name=AZURE_SEARCH_INDEX,
                                    credential=search_creds)
    i = 0
    batch = []
    for s in sections:
        batch.append(s)
        i += 1
        if i % 1000 == 0:
            results = search_client.upload_documents(documents=batch)
            succeeded = sum([1 for r in results if r.succeeded])
            if verbose: print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")
            batch = []

    if len(batch) > 0:
        results = search_client.upload_documents(documents=batch)
        succeeded = sum([1 for r in results if r.succeeded])
        if verbose: print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")

def remove_from_index(filename):
    if verbose: print(f"Removing sections from '{filename.name or '<all>'}' from search index '{AZURE_SEARCH_INDEX}'")
    search_client = SearchClient(endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net/",
                                    index_name=AZURE_SEARCH_INDEX,
                                    credential=search_creds)
    while True:
        filter = None if filename is None else f"sourcefile eq '{filename.name}'"
        r = search_client.search("", filter=filter, top=1000, include_total_count=True)
        if r.get_count() == 0:
            break
        r = search_client.delete_documents(documents=[{ "id": d["id"] } for d in r])
        if verbose: print(f"\tRemoved {len(r)} sections from index")
        # It can take a few seconds for search results to reflect changes, so wait a bit
        time.sleep(2)

def run(filename):
    if removeall:
        remove_blobs(None)
        remove_from_index(None)
    else:
        filename_name = filename
        print("Processing files...")
        if verbose: print(f"Processing '{filename.name}'")
        if remove:
            remove_blobs(filename)
            remove_from_index(filename_name)
            upload_blobs(filename)
            page_map = get_document_text(filename)
            sections = create_sections(filename_name, page_map, use_vectors)
            index_sections(filename_name, sections)
        elif removeall:
            remove_blobs(None)
            remove_from_index(None)

    # Upload the original file into blob storage
    blob_service = BlobServiceClient(account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net", credential=storage_creds)
    cnt_settings = ContentSettings(content_type="application/pdf", content_disposition= "inline")
    blob_container = blob_service.get_container_client(container)
    blob_name = filename.name
    # Read the entire PDF file into a byte stream
    reader = PdfReader(filename)
    pages = reader.pages
    # Create a BytesIO stream to store the merged PDF content
    merged_pdf = BytesIO()
    writer = PdfWriter()

    # Merge all pages into a single PDF
    for i in range(len(pages)):
        writer.add_page(pages[i])

    # Write the merged PDF content to the BytesIO stream
    writer.write(merged_pdf)
    merged_pdf.seek(0)
    blob_container.upload_blob(blob_name, merged_pdf, overwrite=True, content_settings=cnt_settings)

    print(f"File '{blob_name}' uploaded to blob '{blob_name}' in container '{container}'.")
