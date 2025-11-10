import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from chromadb.config import Settings
from dotenv import load_dotenv
import os

load_dotenv()
groq_api = os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')


llm = ChatGroq(model = "llama-3.3-70b-versatile",
               groq_api_key=os.getenv('GROQ_API_KEY'), streaming=True)

txt = TextLoader('FAQ.txt').load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(txt)
embedding= HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
client_settings = Settings(
    persist_directory=None,
    is_persistent=False
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    client_settings=client_settings
)




q_prompt_template = ChatPromptTemplate.from_messages([
    ("system", 
     "Given the chat history and the user's latest question, "
     "rephrase the latest question as a standalone question that can be answered independently."),
    ("user", 
     "Chat history:\n{chat_history}\n\nUser question:\n{question}")
])


qa_prompt_template = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an institute FAQ bot. Answer the question based on the context below. "
     "You can give the additional information if required but make sure it is related with the asked question. "
     "If answer is not present in context, say 'I don't know, would you like me to connect you to a human?'"),
    ("user", 
     "Context:\n{context}\n\nQuestion:\n{question}")
])


retriever = vector_store.as_retriever(search_kwargs={"k": 3})


if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

memory = st.session_state.memory

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    condense_question_prompt=q_prompt_template,
    combine_docs_chain_kwargs={"prompt": qa_prompt_template} 
)





st.title("Institute FAQ Bot")

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role' : 'assistant', 'content' : 'Hi, i am an assistant chatbot of the XYZ institute to help you with the frequently asked questions'}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:=st.chat_input():
    st.session_state.messages.append({'role' : 'user', 'content' : prompt})
    st.chat_message('user').write(prompt)

    chat_history_msgs = []
    for msg in st.session_state['messages']:
        if msg['role'] == 'user':
            chat_history_msgs.append(HumanMessage(content=msg['content']))
        else:
            chat_history_msgs.append(AIMessage(content=msg['content']))

    
    with st.chat_message('assistant'):
        response = qa_chain({"question": prompt,
                            "chat_history": chat_history_msgs})
        st.session_state.messages.append({'role':'assistant', 'content': response['answer']})
        st.write(response['answer'])
        
