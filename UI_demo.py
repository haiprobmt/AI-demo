import streamlit as st
from chat import (
    search_demo, send_message, 
    load_conversation, delete_conversation, 
    get_blob_url_with_sas, upload_to_blob_storage, 
    upload_conversation_to_blob)
import openai
import re
import index_doc

# Initialize conversation
conversation = []
conversation_final = []

st.set_page_config(page_title="Deeeplabs Demo Chatbot", layout="wide")

st.title('Deeeplabs Demo Chatbot')

col1, col2, col3 = st.columns([1, 1, 1])
with col3:
    # Transcription_Mode = st.button('Transcription Mode', use_container_width=True)
    # if Transcription_Mode:
    st.markdown("""
    <a href="https://notebuddyview.z23.web.core.windows.net/" target="_blank">
    <button style='margin: 10px; padding: 10px; background-color: #FFFFFF; color: black; border: curve; cursor: pointer;'>Transcription Mode</button>
    </a>
    """, unsafe_allow_html=True)

logo_url = get_blob_url_with_sas('dl-logo-hamburger.png', "image")
st.sidebar.image(logo_url, width=200)
# with col1:
# Sidebar for system prompt
    # st.sidebar.header("Settings")
st.sidebar.markdown("<h1 style='text-align: left;'>System prompt</h1>", unsafe_allow_html=True)
system_prompt = st.sidebar.text_area(label = "", value = "You are a helpful assistant. Only refer provided source. Do not provide any personal opinions or information. \
                                     Do not provide any medical, legal, financial, or professional advice." , height=200)

# st.sidebar.markdown("<br>"*4, unsafe_allow_html=True)
st.sidebar.markdown("<h1 style='text-align: left;'>Upload File</h1>", unsafe_allow_html=True)
# Upload file
uploaded_file  = st.sidebar.file_uploader("Choose a file", type=["pdf", "docx", "txt"], help="Upload a file to provide context")
# Check if a file was uploaded
if uploaded_file is not None:
    # Save the file to Azure Blob Storage
    file_url = upload_to_blob_storage(uploaded_file)
    # file_name = uploaded_file.name
    index_doc.run(uploaded_file)
    st.sidebar.write('File uploaded successfully!')

# Main layout
# Display conversation
st.header("Conversation")

delete_button = st.button('Clear chat')
if delete_button:
    st.session_state.messages = []
    try:
        st.session_state.messages = []
        delete_conversation("conversation.json")
        delete_conversation("history_json.json")
        st.write('Your chat has been deleted!')
    except:
        st.write('No chat to delete!')

# User input section
st.write(" ")
st.write(" ")

add_source = "\n\nProvide the relevant sourcepage in the end of the response. Do not provide the irrelevant sourcepage, for example: Source: text1.pdf, text2.pdf. Do not provide the sourcepage if the question is generic"
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = []

# User-provided prompt
if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    try:
        history = load_conversation("history_json.json")['history']
        print(history)
    except:
        history = []

    summary_prompt_template = """Below is a summary of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base. Generate a search query based on the conversation and the new question. Source names are not good search terms to include in the search query.

    Summary:
    {summary}

    Question:
    {question}

    Search query:
    """
        
    if len(history) > 0:
        print(summary_prompt_template.format(summary="\n".join(history), question=user_input))
        completion = openai.Completion.create(
            engine='davinci',
            prompt=summary_prompt_template.format(summary="\n".join(history), question=user_input),
            temperature=0.7,
            max_tokens=32,
            stop=["\n"])
        search = completion.choices[0].text
    else:
        search = user_input
    try:
        conversation = load_conversation("conversation.json")
    except:
        conversation = [
                {
                    "role": "system",
                    "content": system_prompt.replace('   ', '')
                }
            ]
    print(search)
    query = search_demo(search)
    conversation.append({"role": "user", "content": query + add_source})
    response = send_message(conversation)
    print(response)
    if "page" in response:
        pattern = r'Source: (.*), page'
        # Find all URLs in the text
        resources = re.findall(pattern, response)
    else:
        pattern = r'Source: (.*)'
        # Find all URLs in the text
        resources = re.findall(pattern, response)
    try:
        if resources[0] == '' or 'N/A' in resources[0]:
            resources_final = []
        else:
            resources_final = list(resources[0].split(", "))
    except:
        resources_final = []
    if "page" in response:
        pattern_1 = r'Source: (.*), page'
        response_final = re.sub(pattern_1, "", response).strip()
    else:
        pattern_1 = r'Source: (.*)'
        response_final = re.sub(pattern_1, "", response).strip()
    conversation[-1]['content'] = user_input
    conversation.append({"role": "assistant", "content": response_final})

    history.append("user: " + user_input)
    history.append("assistant: " + response_final)
    history_json = {"history": history}
    upload_conversation_to_blob("conversation.json", conversation)
    upload_conversation_to_blob("history_json.json", history_json)
    st.session_state.messages.append({"role": "assistant", "content": {"response": response_final, "resources": resources_final}})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        elif message["role"] == "assistant":
            st.write(message["content"]["response"].replace(")", ""))
            resource_list = message["content"]["resources"]
            if len(resource_list) > 0:
                st.write("References:")
                for resource in resource_list:
                    resource.replace(")", "")
                    reference_url = get_blob_url_with_sas(resource, "data-source")
                    st.write(f'[{resource}]({reference_url})')
