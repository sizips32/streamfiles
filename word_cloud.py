import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import PyPDF2
import markdown
from collections import Counter
import pandas as pd
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# NLTK 리소스 다운로드
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

def load_md(file):
    content = file.read().decode("utf-8")
    return markdown.markdown(content)

def load_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

st.title("Word Cloud Generator")

uploaded_file = st.file_uploader("Upload a .md or .pdf file", type=["md", "pdf"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".md"):
        text = load_md(uploaded_file)
    elif uploaded_file.name.endswith(".pdf"):
        text = load_pdf(uploaded_file)

    if text:
        wordcloud = WordCloud(font_path='/Users/soonjaekim/Desktop/AGIs/나눔 글꼴/나눔명조/NanumFontSetup_OTF_MYUNGJO/NanumMyeongjo.otf', width=800, height=400, background_color='white').generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
