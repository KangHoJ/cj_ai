from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

config = PeftConfig.from_pretrained("re2panda/polyglot_1.3B_plain_1104")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/polyglot-ko-1.3b")
model = PeftModel.from_pretrained(model, "re2panda/polyglot_1.3B_plain_1104")

from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import PromptTemplate,  LLMChain, HuggingFacePipeline
import gradio as gr
from newspaper import Article
import re

def crawling(url):
    # url = input('url을 입력하세요')
    article = Article(url,laguage='ko')
    article.download()
    article.parse()
    title = article.title
    text = article.text
    text = '.'.join(article.text.split('.')[:-2:])
    text = ''.join(text.split('기자) ')[1]) # split 할 경우 
    text = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s,.%()]", "", text)
    return title , text

# print("기사내용확인:", crawling('https://www.xportsnews.com/article/1776390'))

def lang_test(url):
    title , text = crawling(url)
    pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 16,
                # do_sample=True,
                top_k=10,       #무작위성 절제
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )
    llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0})
    template = f"""판별은 입력의 기사제목과 기사내용을 분석하여 해당 기사의 낚시성 기사 또는 정상기사, 낚시기사 유형을 출력합니다.
    다음은 기사 제목, 기사 내용를 제공하는 입력과 짝을 이루는 판별 작업을 명령하는 지침입니다. 요청을 적절하게 완료하는 응답을 작성합니다.
    ### 명령:
    주어진 기사를 읽고 낚시성 기사 유무를 판별하라.
    ### 입력:
    기사 제목 : {title} , 기사 내용 :{text}
    ### 판별:
    해당기사는 요약정보는{text}""" 
    
    prompt = PromptTemplate(template=template, input_variables=['title','text'])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain.predict()
    return answer

# lang_test('https://www.xportsnews.com/article/1776390')

iface = gr.Interface(
    fn=lang_test,
    inputs=gr.Textbox(lines=4, placeholder="URL을 입력해주세요"),
    outputs=gr.Textbox(lines=4),
    title="낭만어부 판별 시스템",
    theme='soft',
    description="<div style='font-size: 15px; margin-bottom: 10px;'><strong>기사 URL을 입력하면 낚시 여부를 판별해주는 시스템 입니다.</strong></div>"
               "<ol style='font-size: 15px;'>"
               "<li><strong>기사 URL을 입력하시고 'submit' 버튼을 눌러주세요<strong></li>"
               "<li><strong>낚시기사 판별 여부, 낚시 유형을 확인 가능합니다 (낚시기사가 아닐 시 유형이 나오지 않습니다)<strong></li>"
               "</ol>",
    examples=[["www.naver.com"], 
              ['https://www.xportsnews.com/article/1742486'],
              ['https://www.xportsnews.com/article/1776390']]
)

iface.launch(share=True,inline=True)