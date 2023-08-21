import streamlit as st
import pandas as pd 
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers 
import pandas as pd 

def read_file(file_name : str) -> pd.DataFrame(): 
    # Function to read file
    df_uploader = st.sidebar.file_uploader("Load the file", type = ['csv'])

    if df_uploader is not None : 
        df = pd.read_csv(df_uploader)

        st.session_state['df1'] = df
        return st.session_state['df1']
    else:
        return pd.DataFrame()
    
# set initial configuration
st.set_page_config(
    page_title = "DAX solution",
    layout="wide"
)

if "df1" not in st.session_state : 
    st.session_state['df1'] = pd.DataFrame()


##-------Homepage------##
st.title("QC Solution")

#Read file
df1 = read_file(file_name = "File1")

st.header('Uploaded Data is :')
st.dataframe(st.session_state['df1'], width = 600)

##------Duplicate Check----##
st.write('---')
st.header("Duplicate Check")

df1_duplicate = df1[df1.duplicated(keep = False)]

st.dataframe(df1_duplicate)

##----Outlier Check-----##
st.write("---")
st.header("Outlier Check")

group_col = st.text_input("Goup By Column", "time, sex")
outlier_col = st.text_input("Outlier check column", "tip")


group_col_list = [i.strip() for i in group_col.split(",")]

if st.button("Check Outlier"):
    # run if button is clicked
    df_out = df1.groupby(by = group_col_list)[outlier_col].sum()

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


