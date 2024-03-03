import os
#허깅페이스 개인 APIKey
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub

from crawling import crawling


st.title('질문해보세요')
#허깅페이스에서 모델 불러오기

repo_id = 'EleutherAI/polyglot-ko-1.3b'    

#질문 입력 칸
question = st.text_input('url을 작성해주세요')
st.button("검색", type="primary")

#프롬프트 템플릿
template = """질문:{title}\n 대답:"""
prompt = PromptTemplate(template=template,input_variables=['question'])

#모델 구현
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
)

#LLM Chain 객체 생성
llm_chain = LLMChain(prompt=prompt, llm=llm)
answer = llm_chain.run(question=question)

st.write("정답은", answer)

