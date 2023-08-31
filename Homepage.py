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

st.write("[Sample data for upload](https://github.com/robin991/Quality_Check_Solution/tree/main/data)")

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


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


st.sidebar.button('Clear Chat History', on_click=clear_chat_history)



if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = 'Bot willl generate response'#generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)