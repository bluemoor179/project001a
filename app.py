# Maximum input prompt length for Llama models is 131072 less the number of tokens generated for each task

import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./meta-llama/Meta-Llama-3.1-8B", local_files_only=True)


채월야 = """벽에 걸린 괘종시계가 8시를 알리며 둔한 종소리를 내기 시작했다. 그리고 그와 동시에 문이 열리고 그곳으로 한 남자가 걸어 들어왔다. 전신을 둘러싼 검은 코드 안쪽에 얼핏 비치는 복장은 틀림없는 가톨릭 사제복이다. 비록 술을 금하지는 않지만 경건한 가톨릭의 사제가 직접 바를 찾아온다는 것은 그다지 흔치 않은 일이었다. 더구나 은발의 외국인 젊은이라면."""
print("채월야 토큰 수 : " + str(len(tokenizer.tokenize(채월야))))

검머외 = """
사민당은 칠전팔기 끝에 기어이 독립전쟁의 영웅이자 국공내전의 승전 장수인 박정희 전 국방부 장관을 대선 후보로 내세웠다.

이름만 들어도 대중들의 가슴을 쥐락펴락할 수 있는 독립운동가들 상당수가 고령으로 은퇴하는 이 시기, 일본도 한 자루 들고 왜놈들 모가지를 무수히 썰어댔다는 이 조선 최후의 소드마스터는 독립운동 프리미엄이 굳건히 남아 있는 한국 정계 최고의 불루칩이라고 할 수 있었다.

그리고 이에 맞선 한독당은.
"""
print("검머외 토큰 수 : " + str(len(tokenizer.tokenize(검머외))))


file = open('resources/칠성국/1권/1. 침노.txt', 'r', encoding='UTF8')
칠성국 = file.read()
file.close()
print("칠성국 토큰 수 : " + str(len(tokenizer.tokenize(칠성국))))

file = open('./resources/커피향 나는 열네 번째/001화.txt', 'r', encoding='UTF8')
커피향 = file.read()
file.close()
print("커피향 토큰 수 : " + str(len(tokenizer.tokenize(커피향))))


text = 칠성국
model = ChatOllama(model='llama3.1')
parser = StrOutputParser()
summary_prompt_template = ChatPromptTemplate.from_messages(
    [("system", "너는 글에 대한 특징을 뽑아 설명해주는 어시스턴트이다.")
     , ("user", "이런 글을 읽었습니다."), ("user", "{text}"), ("user", "이 소설은 어떤 특징을 가지고 있다 할 수 있을까요? 이 소설의 장르와  가지고 있는 특징을 알려주세요.")
     , ("ai", "주신 글은 다음과 같은 특징을 가지고 있다 할 수 있습니다.")
    ]
)
summary_chain = summary_prompt_template | model | parser

prompt_template = ChatPromptTemplate.from_messages(
    [("system", "너는 소설의 내용을 이어 써내려가려는 작가이다. 너는 원하는 내용에 대한 글을 어떻게 작성할 수 있는지 시범을 보여주고 있다.")
     , ("user", "다음과 같은 특징을 가진 소설을 써보고 싶은데 어떻게 쓰는게 좋을까요?"), ("user", "특징 : {summary}")
     , ("ai", "그런 특징을 가진 글이라면 다음과 같이 작성해볼 수 있을 것 같습니다."), ("ai", "{text}")
     , ("user", "그 내용에 이어 100 문장 정도 더 작성해주세요.")
     , ("ai", "앞선 내용에서 100 문장 정도 더 작성해보겠습니다.")
    ]
)
chain = prompt_template | model | parser


character_chain = parser

summary = summary_chain.invoke({"text" : text})
print(summary)
print(chain.invoke({"summary" : summary, "text": text}))
