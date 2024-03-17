import streamlit as st
from transformers import pipeline

st.header("Translation from Korean to English")


model = "halee9/translation_ko_en"
translator = pipeline("translation", model=model)

text = "심지어 어떤 문제가 발생할 건지도 어느 정도 예측이 가능하다."
trans = translator(text)

st.subheader(trans[0]["translation_text"])
