"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
from pcconfig import config

import pynecone as pc
from pynecone.base import Base

import os
from datetime import datetime
import json

import openai
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain, LLMChain, ConversationChain, MultiPromptChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessage, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import ZeroShotAgent, Tool



####### GUIDE #######
# 1. 사용할 데이터(project_data_카카오소셜.txt, project_data_카카오싱크.txt, project_data_카카오톡채널.txt)를 preprocessing을 하여 사용한다.
# 2. 데이터는 VectorDB에 업로드하여 사용한다.
# 3. Embedding과 VectorDB를 사용하여 데이터 쿼리를 구성한다.
# 4. LLM과 다양한 모듈을 위해 Langchain 또는 semantic-kernel 둘 중 하나를 사용한다.
# 5. ChatMemory 기능을 사용하여 history를 가질 수 있게 구성한다.
# 6. 서비스 형태는 기본 적인 챗봇 형태로 구성하고, web application을 이용하여 구현한다.
# 7. 최적화를 위해 외부 application을 이용하여 구현해도 된다.(예: web search 기능)
# 8. 다양한 prompt engineering 기법을 활용하여 최대한 일관성 있는 대답이 나오도록 유도한다.
####################


openai.api_key = open('../prj2/prj2/openai_api_key.txt', 'r').read().strip()
os.environ['OPENAI_API_KEY'] = openai.api_key


DATA_PATH = [
    './data/project_data_kakao_social.txt',
    './data/project_data_kakao_sync.txt',
    './data/project_data_kakao_channel.txt',
]
CONFIG = "./prj3/config.json"


def build_db(name: str = 'prj3') -> None:
    """build vectordb with chromaDB. data is loaded from DATA_PATH."""
    db = Chroma(
        name,
        OpenAIEmbeddings(),
        persist_directory='./db',
    )
    if os.path.exists(os.path.join('./db', 'chroma.sqlite3')):
        return db

    for fname in DATA_PATH:
        data = TextLoader(fname).load()
        spliter = CharacterTextSplitter(chunk_size=300, chunk_overlap = 50)
        documents = spliter.split_documents(data)
        db.add_documents(documents)
    return db


def get_search(query) -> str:
    search = GoogleSearchAPIWrapper(
        google_api_key=os.getenv("GOOGLE_API_KEY","AIzaSyD9SMbVosJIV0-6bktmlVJSdh7Zm7WA1HU"),
        google_cse_id=os.getenv("GOOGLE_CSE_ID","d05a99d35d71042a6")
    )
    search_tool = Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    )
    return search_tool.run(query)
    


def build_chains(memory) -> dict:
    chains = {}
    '''build chains with qa and llm.'''
    # Downstream chains
    db = build_db()
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
        open(opt['router_info']['chains']['chatbot']['prompt_template_path']).read().strip()
    )
    # chatbot
    chains['chatbot'] = ConversationalRetrievalChain(
        question_generator=LLMChain(
            llm=ChatOpenAI(),
            prompt=CONDENSE_QUESTION_PROMPT,
            verbose=True,
        ),  # make question from query, and then query to db
        retriever=db.as_retriever(),
        combine_docs_chain=load_qa_chain(llm=ChatOpenAI(temperature=0), chain_type='map_reduce', verbose=True),  # get answer from db and make a response
        memory=memory,
        output_key="answer",
        verbose=True,
    )  # https://python.langchain.com/docs/use_cases/question_answering/integrations/openai_functions_retrieval_qa#using-in-conversationalretrievalchain

    chains['search'] = LLMChain(
        llm=ChatOpenAI(),
        prompt=PromptTemplate.from_template(
            open(opt['router_info']['chains']['search']['prompt_template_path']).read().strip(),
        ),
        memory=memory,
        output_key="answer",
        verbose=True,
    )

    chains['default'] = ConversationChain(
        llm=ChatOpenAI(),
        input_key='query',
        output_key='answer',
        prompt=PromptTemplate.from_template(
            '''
                The following is a friendly conversation between a human and an AI.
                The AI is talkative and provides lots of specific details from its context.
                If the AI does not know the answer to a question, it truthfully says it does not know.
                \n\nCurrent conversation:\n{chat_history}\nHuman: {query}\nAI:
            '''
        ),
        memory=memory,
        verbose=True
    )

    # Router
    destinations = [
        f'{name}: {chain_info["description"]}' for name, chain_info in opt['router_info']['chains'].items()
    ]
    destinations_str = '\n'.join(destinations)
    router_template = open(opt['router_info']['prompt_template_path']).read().strip().format(
        destinations=destinations_str
    )  # change from (langchain.chains.router.multi_prompt_prompt) MULTI_PROMPT_ROUTER_TEMPLATE
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["query"],
        # output_parser=StrOutputParser(),  # automatically route to the correct chain, not string
    )
    chains['router'] = LLMChain(  # why not LLMRouterChain?
        llm=ChatOpenAI(),
        prompt=router_prompt,
        memory=memory,
        verbose=True
    )  # Convenience constructor (you can also use the LLMRouterChain constructor directly)

    # multi-chain
    '''  # cannot use ConversationRetrievalChain with destination_chains
    # total_chain = MultiPromptChain(
        router_chain=router_chain,
        destination_chains={
            'chatbot': chatbot_chain,
            'search': search_chain,
        },
        default_chain=default_chain,
        verbose=True,
    )
    '''
    return chains

    
def query(query: str, chains, memory) -> str:
    target_chain = json.loads(chains['router'].run(query=query).replace('json', '').replace('```', '').replace('"""','').strip())['destination']
    if target_chain == 'chatbot':
        ret = chains['chatbot'].run(question=query)
    elif target_chain == 'search':
        ret = chains['search'].run(query=query, web_search_results=get_search(query))
    else:
        ret = chains['default'].run(query=query)
    if isinstance(ret, str):
        return ret
    elif isinstance(ret, dict):
        return ret.get('answer', 'Nothing to say.')
    return 'Nothing to say.'


class Message(Base):
    query: str
    answer: str
    created_at: str
    name: str = "kakao chatbot"


class State(pc.State):
    """The app state."""

    text: str = ""
    name: str = "kakao chatbot"
    messages: list[Message] = [
        Message(
            query="",
            answer="안녕하세요, 챗봇서비스를 시작합니다. 궁금한 내용을 물어보세요.",
            created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
            name="kakao chatbot",
        )
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        self.chains = build_chains(self.memory)

    @pc.var
    def output(self) -> str:
        if not self.text.strip():
            return "Answer will appear here."
        # return query_db(self.text, qa)
        return query(self.text, self.chains, self.memory)

    def post(self):
        self.messages = [
            Message(
                query=self.text,
                answer=self.output,
                created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
                name=self.name,
            )
        ] + self.messages


def header():
    """Basic instructions to get started."""
    return pc.box(
        pc.text("KAKAO CHATBOT", font_size="2rem"),
        pc.text(
            "Ask anything about Kakao Services!",
            margin_top="0.5rem",
            color="#666",
        ),
    )


def text_box_from_me(text):
    return pc.text(
        text,
        background_color="#fef01b",
        padding="1rem",
        border_radius="8px",
        # right align
        align_self="flex-end",
        display="none" if not text or text == 'None' else "inline-block",
    )


def text_box_from_bot(text):
    return pc.text(
        text,
        background_color="#fff",
        padding="1rem",
        border_radius="8px",
        # left align
        align_self="flex-start",
    )


def message(message):
    return pc.box(
        pc.vstack(
            text_box_from_me(message.query),
            text_box_from_bot(message.answer),
            pc.box(
                pc.text(message.name),
                pc.text(" · ", margin_x="0.3rem"),
                pc.text(message.created_at),
                display="flex",
                font_size="0.8rem",
                color="#666",
            ),
            spacing="0.3rem",
            align_items="left",
        ),
        background_color="#f5f5f5",
        padding="1rem",
        border_radius="8px",
    )


def index() -> pc.Component:
    """The main view."""
    return pc.container(
        header(),
        pc.input(
            placeholder="Type something...",
            on_blur=State.set_text,
            margin_top="1rem",
            border_color="#eaeaef"
        ),
        # button places right side of container
        pc.button("Send", on_click=State.post, margin_top="1rem", float="right"),
        pc.vstack(
            pc.foreach(State.messages, message),
            margin_top="2rem",
            spacing="1rem",
            align_items="left"
        ),
        padding="2rem",
        max_width="600px"
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index)
opt = json.load(open(CONFIG, 'r'))
app.compile()
