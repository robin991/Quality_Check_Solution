import streamlit as st
import pandas as pd 
#from langchain.prompts import PromptTemplate
#from langchain.llms import CTransformers 
#import pandas as pd 

def read_file(file_name : str) -> pd.DataFrame(): 
    # Function to read file
    df_uploader = st.sidebar.file_uploader("Load the file", type = ['csv'])

    if df_uploader is not None : 
        df = pd.read_csv(df_uploader)

        st.session_state['df'] = df
        return st.session_state['df']
    else:
        return pd.DataFrame()

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]   

    
# set initial configuration
st.set_page_config(
    page_title = "DAX solution",
    layout="wide"
)

if "df" not in st.session_state : 
    st.session_state['df'] = pd.DataFrame()
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

##-------Homepage------##
st.title("QC Solution")

#Read file from side bar
df = read_file(file_name = "File1")

st.sidebar.write("[Sample file download link](https://github.com/robin991/Quality_Check_Solution/tree/main/data)")


st.header('Uploaded Data is :')
st.dataframe(st.session_state['df'], width = 600)

##------Duplicate Check----##
# st.write('---')
# st.header("Duplicate Check")

# df_duplicate = df[df.duplicated(keep = False)]

# st.dataframe(df_duplicate)

##----Outlier Check-----##
st.write("---")
st.header("Outlier Check")

group_col = st.text_input("Goup By Column", "time, sex")
outlier_col = st.text_input("Outlier check column", "tip")


group_col_list = [i.strip() for i in group_col.split(",")]

#if st.button("Check Outlier"):
# run if button is clicked
df_out = df.groupby(by = group_col_list)[outlier_col].sum()

df_out = df_out.reset_index()
st.dataframe(df_out)

st.write(f"This is the mean :{round(df_out.loc[:,outlier_col].mean(),3)}")
st.write(f"This is the std :{round(df_out.loc[:,outlier_col].std(),3)}")

# threshold calculation
thresh = 2*round(df_out.loc[:,outlier_col].std(),3)
st.write(f"This is the outlier thershold(+/-) :{thresh}")

st.write("Outlier values:")

# Final Outlier dataframe
df_out = df_out[(df_out[outlier_col] <-thresh) | (df_out[outlier_col] >thresh) ]

st.dataframe(df_out)

st.write("---")
#with st.expander('Chat Bot'):
st.write('analysising below two dataframes (duplicate and outlier)')
col1, col2 = st.columns(2)
#col1 = st.dataframe(df_duplicate)
col2 = st.dataframe(df_out)

## ------chat bot -----##
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain 

# store embeddings
DB_FAISS_PATH = "vectorestore/db_faiss"

# loading the model
def load_llm():
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type = "llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

st.title("Chat with CSV using LLAMAv2 quantized")

# build sidebar to upload the file on streamlit
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type = "csv")

if uploaded_file :

    
    
    # create a temporary file object
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                'delimiter': ','}) # csv loader needs  a filepath hence we created a temporary file path

    # data has the csv
    data = loader.load()
    #st.json(data)

    # word embedding model ( vector creation)
    embeddings = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs = {'device' : 'cpu'})
    
    db = FAISS.from_documents(data,embeddings)
    # save the db to the path
    db.save_local(DB_FAISS_PATH)

    # load llm model . it will be passed in conversation retrieval chain
    llm = load_llm()

    #chain call
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    # function for streamlit chat
    def conversational_chat(query):
        result = chain({'question':query, "chat_history" : st.session_state['history']})
        st.session_state['history'].append([query, result['answer']])
        return result['answer']
    
    if 'history' not in st.session_state : 
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ['Hello, Ask me anything about' + uploaded_file.name]
    
    if 'past' not in st.session_state:
        st.session_state['past'] = ['Hey! ']

    # assigning containers for the chat history

    response_container = st.container()
    
    container = st.container()

    with container:
        with st.form(key = "my_form", clear_on_submit = True):
            user_input = st.text_input("Query:", placeholder = "Talk to your CSV Data here", key = 'input')

            submit_buttom = st.form_submit_button(label ='Send')

        if submit_buttom and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i],
                        is_user = True,
                        key = str(i) + '_user',
                        avatar_style = "big-smile")
                message(st.session_state["generated"][i],
                        key = str(i) ,
                        avatar_style = "thumbs")