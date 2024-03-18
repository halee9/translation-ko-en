import streamlit as st
from transformers import pipeline

st.header("Translation from Korean to English")


# model = "halee9/translation_ko_en"
# translator = pipeline("translation", model=model)

# text = "심지어 어떤 문제가 발생할 건지도 어느 정도 예측이 가능하다."
# trans = translator(text)
# st.subheader(trans[0]["translation_text"])

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("halee9/translation_en_ko")
model = AutoModelForSeq2SeqLM.from_pretrained("halee9/translation_en_ko")

input_text = [
    "어떤 학문이든지 일정의 성취를 이루기 위해서는 끊임없는 반복이 필요하다.",
    "이날 비대위가 사직서 제출일로 결의한 25일은 행정처분 사전통지를 받은 전공의들이 의견 제출을 해야 하는 마지막 날이다."
]
max_token_length = 128
inputs = tokenizer(input_text, return_tensors="pt",
                   padding=True, max_length=max_token_length)

koreans = model.generate(
    **inputs,
    max_length=max_token_length,
    num_beams=5,
)

st.subheader("Done")