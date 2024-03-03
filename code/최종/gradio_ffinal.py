from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import PromptTemplate,  LLMChain, HuggingFacePipeline
import gradio as gr
from newspaper import Article
import re
from summa.summarizer import summarize
import pandas as pd
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import CountVectorizer
from summa.summarizer import summarize
config = PeftConfig.from_pretrained("re2panda/polyglot_1.3B_plain_1104")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/polyglot-ko-1.3b")
model = PeftModel.from_pretrained(model, "re2panda/polyglot_1.3B_plain_1104")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")


def crawling(url): # 엑스포츠
    # url = input('url을 입력하세요')
    article = Article(url,laguage='ko')
    article.download()
    article.parse()
    title = article.title
    text = article.text
    text = '.'.join(article.text.split('.')[:-2:])  # 일반 기사로 할시 지움(엑스포츠 기사만 적용)
    text = ''.join(text.split('기자) ')[1]) # 일반 기사로 할시 지움(엑스포츠 기사만 적용)
    text = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s,.%()]", "", text)
    text_sum = summarize(text,ratio=0.2)
    # text_sum = '\n'.join(summarize(text,word_coutn=200).split('.'))
    return title , text , text_sum

kiwi = Kiwi()
def extract_nouns(text): # 토큰추출
        result = kiwi.tokenize(text)
        for token in result:
            if token.tag in ['NNG', 'NNP']:
                yield token.form


# print("기사내용확인:", crawling('https://www.xportsnews.com/article/1776390'))

def lang_test(url):
    title , text , text_sum = crawling(url)
    data = {'text': [text]}
    df = pd.DataFrame(data)
    cv = CountVectorizer(tokenizer=extract_nouns, min_df=1)
    dtm = cv.fit_transform(df.text) 
    word_count = pd.DataFrame({'word': cv.get_feature_names_out(),'count': dtm.sum(axis=0).flat}) #빈도수 만들기
    top5_keywords = word_count.sort_values('count', ascending=False).head(5).reset_index(drop=True) # 빈출 빈도수 
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
    해당기사는""" 
    
    prompt = PromptTemplate(template=template, input_variables=['title','text'])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain.predict() 
    # answer = llm_chain.predict()
    # if '낚시성기사입니다' in answer:
    #     answer
    # else:
    #     answer = llm_chain.predict() + f'\n\n <본문 요약> \n {text_sum}'  
    return answer , text_sum , top5_keywords

# lang_test('https://www.xportsnews.com/article/1776390')

iface = gr.Interface(
    fn=lang_test,
    inputs=gr.Textbox(lines=2, placeholder="URL을 입력해주세요"),
    outputs=[gr.Textbox(lines=2,label='낚시성 판별 결과'),gr.Textbox(label="요약 정보"),gr.Textbox(label="빈출 빈도 수")],
    title = "<center><img src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F90XMW%2FbtszQA0psbQ%2Fg0SVPwFcNL017I6cKCWjhk%2Fimg.png' width=500 ></center>",
    theme='soft',
    description=
                "<center><div style='font-size: 20px; margin-bottom: 30px;'><strong>진실을 찾아가는 PressPulse, 언론의 건강상태를 체크하세요.</strong></div></center>"
                "<div style='font-size: 15px; margin-bottom: 10px;'><strong>기사 URL을 입력하면 낚시 여부를 판별해주는 시스템 입니다.</strong></div>"
                "<ol style='font-size: 15px;'>"
                "<li><strong>기사 URL을 입력하시고 'submit' 버튼을 눌러주세요<strong></li>"
                "<li><strong>낚시기사 판별 여부, 낚시 유형을 확인 가능합니다 (낚시기사가 아닐 시 유형이 나오지 않습니다)<strong></li>"
                "</ol>",
    examples=[["https://www.xportsnews.com/article/1715199"], 
              ['https://www.xportsnews.com/article/1742486'],
              ['https://www.xportsnews.com/article/1776390']]
)
# iface.launch(inline=True)
iface.launch(share=True,inline=True)