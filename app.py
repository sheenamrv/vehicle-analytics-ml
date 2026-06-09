import streamlit as st
from src.test import *

st.title("Testing Windows Executable")
st.write("By: Andrianna")

name = st.text_input("Enter your name: ", placeholder="John Doe")

if st.button("Greet me!"):
    st.write(greet(name))