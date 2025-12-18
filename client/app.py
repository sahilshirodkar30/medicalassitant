import streamlit as st
from component.upload import render_uploader
from component.history_download import render_history_download
from component.chatUI import render_chat


st.set_page_config(page_title="AI Medical Assistant",layout="wide")
st.title(" 🩺 Medical Assistant Chatbot")


render_uploader()
render_chat()
render_history_download()