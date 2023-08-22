import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplate import css,bot_template,user_template

def get_pdf_text(pdf_docs):
    text="" #raw text.Storing all of text from pdfs
    
    #loop over pdf i povezati sa text varijablom
    for pdf in pdf_docs:
        
        #Create pdf object that have pages for each pdfs
        pdf_reader=PdfReader(pdf)
        
        #loop over pages
        for page in pdf_reader.pages:
            text+=page.extract_text()
    # single string with all of text
    return text
    
def get_text_chuncks(raw_text):
    text_splitter=CharacterTextSplitter(     
        separator="\n",
        chunk_size=1000,#koliko ce rijeci uzest za jedan chunck
        chunk_overlap=200, # ako stane na pola rijeci uzest ce prethodnih 200 da budemo sigurni da ih je sve obuhvatio
        length_function=len
        )

    chunks=text_splitter.split_text(raw_text)
    return chunks
    
    
def get_vectorestore(text_chunks):
    embeddings=OpenAIEmbeddings(model_name="hkunlp/instructor-xl")
    vectorestore=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorestore
    
    #Generate new messages
def get_conversation_chain(vectorstore):
    # Memory 
    llm=ChatOpenAI()
    memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
        
    )
    return conversation_chain
    
def  handle_userinput(user_question):
    
    #sadsrzi sve konfiguracije iz nase VB i memory
    #Ako ga budemo koristili i nastavili on ce zapamtiti prijasnje pitanje 
    
    response=st.session_state.conversation_chain({'question':user_question})
    #New session state for history
    st.session_state.chat_history=response['chat_history']
    
    #Loop throug chat history with index and content of that index
    for i ,message in enumerate(st.session_state.chat_history):
        if i % 2==0:
            st.write(user_template.replace('{{MSG}}',message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace('{{MSG}}', message.content),unsafe_allow_html=True)
        
    
def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with multiple PDFs',page_icon=":books:")
    
    st.write(css,unsafe_allow_html=True)
    
    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain =None
        
    if 'chat_history' not in st.session_state:
       st.session_state.chat_history =None
            
            
    st.header("Chat with multiple PDFs :books:")
    user_question=st.text_input("Ask question about your docs:")
    
    if user_question:
        handle_userinput(user_question)
        
        
    st.write(user_template.replace('{{MSG}}','Hello robot'),unsafe_allow_html=True)
    st.write(bot_template.replace('{{MSG}}', 'Hello human'),unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("Your docs")
        pdf_docs=st.file_uploader("Upload your PDFs here and click on 'Proces'",accept_multiple_files=True)
        

        
        if st.button("Process"):# kad se dogodi button onda reloada sve
            
           with st.spinner("Processing"):
                # get pdf text
                
                #return all of the pdf content
                raw_text=get_pdf_text(pdf_docs) 
                st.write(raw_text)
                
                
                
                # get the text chuncks.Return list of chunck of text which we ll feed our VB
                text_chuncks=get_text_chuncks(raw_text)
                st.write(text_chuncks)
                
                # create vectore store with embedings
                vectorestore=get_vectorestore(text_chuncks)
                
                # Instance of conversation chain
                
                st.session_state.conversation_chain=get_conversation_chain(vectorestore)#Conversation chain je linkan na sesiju i onda streamlit zna da ovo netreba reinicijalizirati

                
        

if __name__ == '__main__':
    main()