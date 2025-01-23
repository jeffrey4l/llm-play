#!/bin/python
# -*- coding: utf-8 -*-

# learn from https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-1-build-a-basic-chatbot

import pprint
import argparse
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.chat_models import ChatZhipuAI
from IPython.display import display, Image
from langchain_core.globals import set_debug
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.runnables import RunnableConfig
import logging

LOG = logging.getLogger(__name__)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def get_llm(conf):
    if conf.model == 'deepseek':
        key = os.environ.get("DEEPSEEK_API_KEY")
        return ChatOpenAI(
            model='deepseek-chat',
            openai_api_base='https://api.deepseek.com',
            openai_api_key=key,
            max_tokens=4096)
    elif conf.model == 'zhipu':
        api = os.environ.get("ZHIPUAI_API_KEY")
        llm = ChatZhipuAI(
            api_key=api,
            model="glm-4-plus",
            temperature=0
        )
        return llm
    raise ValueError("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', '-m',
        type=str, required=True,
        choices=['deepseek', 'zhipu'],
        default='zhipu')
    parser.add_argument('--debug', '-d', action='store_true')
    conf = parser.parse_args()
    if conf.debug:
        set_debug(True)
    logging.basicConfig(
        level=logging.DEBUG if conf.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s.%(funcName)s:%(lineno)d %(message)s"
    )
    llm = get_llm(conf)
    graph_builder = StateGraph(State)

    config: RunnableConfig = {"configurable": {"thread_id": "123"}}

    def chatbot(state: State):
        LOG.error("state: %s", state)
        return {"messages": [llm.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    memory = MemorySaver()

    app = graph_builder.compile(checkpointer=memory)
    with open('a.png', 'wb') as f:
        f.write(app.get_graph().draw_mermaid_png())

    def stream_graph_updates(user_input: str):
        for event in app.stream({"messages": [{"role": "user", "content": user_input}]}):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)

    def stream_graph_updates2(user_input: str):
        state = app.get_state(config).values
        messages = []
        if state.get("messages") is None:
            messages.append(SystemMessage("能精准告知食物热量，对非食物信息礼貌辨别"))
        messages.append(HumanMessage(user_input))
        trimed_messages = trim_messages(
            messages,
            include_system=True,
            token_counter=len,
            max_tokens=2,
            strategy="last")
        LOG.info("messages: %d, trimmed_messages: %d" % (len(messages), len(trimed_messages)))
        inputs = {"messages": trimed_messages}
        for msg, metadata in app.stream(inputs, config, stream_mode="messages"):
            if isinstance(msg, AIMessageChunk):
                print(msg.content, end="", flush=True)
                if msg.response_metadata:
                    print('')
                    print(msg.response_metadata)
            else:
                print(msg, metadata)
        print()

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates2(user_input)
        state = app.get_state(config).values
        pprint.pprint(state)


if __name__ == "__main__":
    main()
