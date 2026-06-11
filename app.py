import streamlit as st
import pages.load_page.load_data as load_data
from src.data.test_load import *

st.title("Testing Windows Executable")
st.write("By: Andrianna")

load_data.load_page()