import os
#허깅페이스 개인 APIKey
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
import gradio as gr
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from crawling import crawling


def update(url):
    title , text = crawling(url)  
    repo_id = 'EleutherAI/polyglot-ko-1.3b'
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64})
    template = f"질문: {title} 은 기사 제목이랑 어울릴까?\n대답:"
    prompt = PromptTemplate(template=template, input_variables=['title'])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain.run(question="")
    return answer


iface = gr.Interface(
    fn=update,
    inputs=gr.Textbox(lines=3,placeholder="URL을 입력해주세요"),
    outputs=gr.Textbox(lines=3),
    title="낭만어부 판별 시스템",
    theme='soft',
    description="URL을 입력하고 '제출' 버튼을 누르세요.",
    examples=[["www.naver.com"], 
              ['https://www.xportsnews.com/article/1742486'],
              ['https://www.xportsnews.com/article/1776390']] 
    
)


iface.launch()
