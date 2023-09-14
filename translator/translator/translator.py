"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

# Import pynecone.
import openai
import os
from datetime import datetime

import pynecone as pc
from pynecone.base import Base


openai.api_key = open('./translator/openai_api_key.txt', 'r').read().strip()


role = {
    "simsim2": 
        '''
            Japanese who never lets the other person get bored.
            Very witty and responsive.
            Uses half-speech and talks in a friendly manner.
            Say only in Japanese.
        ''',
    "2ruda": 
        '''
            Korean woman in her 20s. She has a sweet and affectionate way of speaking and smiles a lot.
            Say only in Korean.
        ''',
    "sangdam1": 
        '''
            Help desk staff. Very polite, courteous, and factual.
            Say only in English.
        ''',
}


def text_to_chatgpt(text: str, name: str) -> str:
    # fewshot 예제를 만들고
    '''
    def build_fewshot(src_lang, trg_lang):
        fewshot_messages = []

        for src_text, trg_text in zip(src_examples, trg_examples):
            fewshot_messages.append({"role": "user", "content": src_text})
            fewshot_messages.append({"role": "assistant", "content": trg_text})

        return fewshot_messages
    '''

    # system instruction 만들고
    system_instruction = f"assistant's name is {name}, who is {role[name]}"

    # messages를만들고
    # fewshot_messages = build_fewshot(src_lang=src_lang, trg_lang=trg_lang)

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": text}
    ]

    # API 호출
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages=messages)
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
    name: str = "2ruda"

    @pc.var
    def output(self) -> str:
        if not self.text.strip():
            return "Answer will appear here."
        translated = text_to_chatgpt(
            self.text, name=self.name)
        return translated

    def post(self):
        self.messages = [
            Message(
                original_text=self.text,
                text=self.output,
                created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
                name=self.name,
            )
        ] + self.messages


# Define views.


def header():
    """Basic instructions to get started."""
    return pc.box(
        pc.text("Translator 🗺", font_size="2rem"),
        pc.text(
            "Translate things and post them as messages!",
            margin_top="0.5rem",
            color="#666",
        ),
    )


def down_arrow():
    return pc.vstack(
        pc.icon(
            tag="arrow_down",
            color="#666",
        )
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


def smallcaps(text, **kwargs):
    return pc.text(
        text,
        font_size="0.7rem",
        font_weight="bold",
        text_transform="uppercase",
        letter_spacing="0.05rem",
        **kwargs,
    )


def output():
    return pc.box(
        pc.box(
            smallcaps(
                "Output",
                color="#aeaeaf",
                background_color="white",
                padding_x="0.1rem",
            ),
            position="absolute",
            top="-0.5rem",
        ),
        pc.text(State.output),
        padding="1rem",
        border="1px solid #eaeaef",
        margin_top="1rem",
        border_radius="8px",
        position="relative",
    )


def index():
    """The main view."""
    return pc.container(
        header(),
        pc.input(
            placeholder="Type something...",
            on_blur=State.set_text,
            margin_top="1rem",
            border_color="#eaeaef"
        ),
        pc.select(
            list(role.keys()),
            value=State.name,
            placeholder="Select name of bot.",
            on_change=State.set_name,
            margin_top="1rem",
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
app.add_page(index, title="Translator")
app.compile()