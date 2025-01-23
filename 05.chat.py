#!/usr/bin/env python3

import logging
import os

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.callbacks import BaseCallbackHandler


LOG = logging.getLogger(__name__)


class ConsoleSyncHandler(BaseCallbackHandler):

    def on_llm_new_token(self, token: str, *args, **kwargs):
        print(f'{token}', end='', flush=True)
        super().on_llm_new_token(token, *args, **kwargs)

    def on_chain_start(self, *args, **kwargs):
        LOG.info("同步调用: chain start")
        super().on_chain_start(*args, **kwargs)

    def on_chain_end(self, *args, **kwargs):
        print('')
        LOG.info("同步调用: chain end")
        super().on_chain_end(*args, **kwargs)



def main():
    api = os.environ.get("ZHIPUAI_API_KEY")
    logging.basicConfig(level=logging.INFO)
    # change memory to BufferMemory
    ai = ChatZhipuAI(
        api_key=api,
        model="glm-4-plus",
        temperature=0.1, streaming=True, callbacks=[ConsoleSyncHandler()],
    )
    template = """You are a chatbot having a conversation with a human.

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"], template=template
    )

    p = prompt | ai
    p.invoke({"chat_history": "你好", "human_input": "你好"})


if __name__ == "__main__":
    main()
