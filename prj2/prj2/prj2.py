"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
import os
from pcconfig import config
from datetime import datetime

import pynecone as pc
from pynecone.base import Base

import openai

from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


openai.api_key = open('./prj2/openai_api_key.txt', 'r').read().strip()
os.environ['OPENAI_API_KEY'] = openai.api_key


DATA_PATH = './prj2/data.txt'
ROLE = 'You are a chatbot which can describe kakao API.'


def load_data() -> None:
    """Load data from file."""
    data = TextLoader(DATA_PATH).load()
    spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap = 200)
    documents = spliter.split_documents(data)
    db = Chroma.from_documents(documents, OpenAIEmbeddings())
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0), chain_type="stuff", retriever=db.as_retriever())
    return qa


def query_db(query: str, qa) -> str:
    return qa.run(query)


def text_to_chatgpt(text: str) -> str:
    messages = [
        {"role": "system", "content": ROLE},
        {"role": "user", "content": text}
    ]

    # API 호출
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages=messages,
                                            max_tokens=2048)
    answer = response['choices'][0]['message']['content']
    # Return
    return answer


class Message(Base):
    original_text: str
    text: str
    created_at: str
    name: str


class State(pc.State):
    """The app state."""

    text: str = ""
    messages: list[Message] = []
    name: str = "kakao chatbot"

    @pc.var
    def output(self) -> str:
        if not self.text.strip():
            return "Answer will appear here."
        return query_db(self.text, qa)

    def post(self):
        self.messages = [
            Message(
                original_text=self.text,
                text=self.output,
                created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
                name=self.name,
            )
        ] + self.messages


def header():
    """Basic instructions to get started."""
    return pc.box(
        pc.text("KAKAO CHATBOT", font_size="2rem"),
        pc.text(
            "Ask anything about Kakao Sync!",
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
            text_box_from_me(message.original_text),
            text_box_from_bot(message.text),
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
app.add_page(index, title='Kakao Chatbot')
qa = load_data()
app.compile()
