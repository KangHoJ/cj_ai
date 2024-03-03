from newspaper import Article
import re 


def crawling(url):
    url = input('url을 입력하세요')
    article = Article(url,laguage='ko')
    article.download()
    article.parse()
    title = article.title
    text = article.text
    text = '.'.join(article.text.split('.')[:-2:])
    text = ''.join(text.split('기자) ')[1]) # split 할 경우 
    text = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s,.%()]", "", text)
    return title , text
    # text = re.search(r'기자\)\s*(.*)', text)[1] # 정규표현식으로 할경우

