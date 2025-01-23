#!/usr/bin/env python3

import argparse
import queue
import threading
import logging
import os
import time
from langchain_openai import ChatOpenAI

from chromadb import config
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
import numpy as np
from rich.console import Console
from rich.markdown import Markdown
import streamlit as st

from chromadb.config import Settings
from langchain_core.callbacks import BaseCallbackHandler

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline


LOG = logging.getLogger(__name__)

api = os.environ.get("ZHIPUAI_API_KEY")


class BaseAction:
    name = ''
    def add_parser(self, parser: argparse.ArgumentParser):
        pass

    def run(self, conf: argparse.Namespace):
        pass


class UIAction(BaseAction):
    name = 'ui'

    def __init__(self):
        pass

    def add_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument('--data', type=str, required=True)
        ChatFactory.add_parser(parser)
        EmbeddingFactor.add_parser(parser)

    def gen_awnser(self, question, conf, embedding):
        db = Chroma(
            persist_directory=conf.data,
            embedding_function=embedding)
        LOG.info(f"database info: {db._collection.count()}")

        a = queue.Queue()

        class CallBack(BaseCallbackHandler):

            def __init__(self, queue):
                self.queue = queue

            def on_chain_start(self, *args, **kwargs):
                print("on chain start")

            def on_chain_end(self, *args, **kwargs):
                print(args, kwargs)
                print("on chain end")

            def on_llm_new_token(self, token: str, *args, **kwargs):
                self.queue.put(token)
                super().on_llm_new_token(token, *args, **kwargs)

        chat = ChatFactory.get_chat(conf.model, callbacks=[CallBack(a)])
        template = """
你是一个IaaS云系统AI助手，回答使用以下上下文来回答最后的问题;
如果你不知道答案，就说你不知道，不要试图编造答案;
描写出详细的步骤;

上下文内容：{context}

问题: {question}
"""

        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template)
        qa_chain = RetrievalQA.from_chain_type(
            chat,
            retriever=db.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

        # qa_chain.stream({"query": question})
        def w():
            result = qa_chain.invoke({"query": question})
            a.put('\n### 相关资料:\n')
            for d in result['source_documents']:
                a.put(' * %s\n' % d.metadata['source'])
            a.put(None)

        threading.Thread(target=w).start()
        messages = ''
        while True:
            try:
                item = a.get(timeout=1)
                if item is None:
                    break
                yield item
                messages += item
            except queue.Empty:
                continue

    def run(self, conf: argparse.Namespace):
        st.set_page_config(layout="wide")
        st.markdown("<h1 style='margin-top: 0;'>用户手册问答助手</h1>", unsafe_allow_html=True)

        # Add custom CSS for auto height and width
        st.markdown(
            """
            <style>
            .stContainer {
                height: auto !important;
                width: auto !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        if 'embedding' not in st.session_state:
            st.session_state.embedding = EmbeddingFactor.get_embedding(conf)

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        messages = st.container()  # Removed height parameter

        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])

        if prompt := st.chat_input("say"):
            st.session_state.messages.append({"role": "user", "text": prompt})
            # Display the user's message immediately
            messages.chat_message("user").write(prompt)
            print(f'用户问题： {prompt}')

            answer = self.gen_awnser(prompt, conf, st.session_state.embedding)
            def gen():
                st.session_state.messages.append({"role": "assistant", "text": ''})
                for a in answer:
                    print(f'{a}', end='', flush=True)
                    st.session_state.messages[-1]['text'] += a
                    yield a
            messages.chat_message("assistant").write_stream(gen)


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


class AskAction(BaseAction):
    name = 'ask'

    def add_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument('--data', type=str, required=True)
        parser.add_argument('question', type=str)
        parser.add_argument('--with-docs', action='store_true')
        ChatFactory.add_parser(parser)
        EmbeddingFactor.add_parser(parser)

    def run(self, conf):
        db = Chroma(
            persist_directory=conf.data,
            embedding_function=EmbeddingFactor.get_embedding(conf))
        LOG.info(f"database info: {db._collection.count()}")

        chat = ChatFactory.get_chat(conf.model, callbacks=[ConsoleSyncHandler()])
        template = """你是一个IaaS云系统AI助手，回答使用以下上下文来回答最后的问题;如果你不知道答案，就说你不知道，不要试图编造答案;使用标准的 markdown 格式进行回答；

上下文内容
{context}

问题:
{question}
"""
        console = Console()
        console.print(f'问题: {conf.question}')

        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template)
        qa_chain = RetrievalQA.from_chain_type(
            chat,
            retriever=db.as_retriever(),
            return_source_documents=True,
            callbacks=[ConsoleSyncHandler()],
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
        result = qa_chain.invoke({"query": conf.question})

        r = '\n### 相关资料:\n\n'

        for d in result['source_documents']:
            r += ' - %s\n' % d.metadata['source']
        print(r)

        # d = Markdown(r)
        # console.print(d)
        if conf.with_docs:
            print("参考文档:")
            for doc in result['source_documents']:
                d = Markdown(doc.page_content)
                print("------------------------------------------")
                console.print(d)


class EmbeddingFactor:

    @staticmethod
    def add_parser(parser: argparse.ArgumentParser):
        parser.add_argument('--embedding', type=str, required=True)

    @staticmethod
    def get_embedding(conf):
        LOG.info("Reload the embeddings")
        if conf.embedding == 'zhipu':
            return ZhipuAIEmbeddings(api_key=api)
        elif conf.embedding == 'hugeface':
            return HuggingFaceEmbeddings(
                model_name="jinaai/jina-embeddings-v3",
                model_kwargs={
                    "trust_remote_code": True,
                }
            )
        return ValueError("unknown embedding type: %s" % conf.embedding)


class ChatFactory:
    @staticmethod
    def add_parser(parser: argparse.ArgumentParser):
        parser.add_argument("--model", type=str, required=True)

    @classmethod
    def get_chat(cls, model, callbacks=None):
        if model == 'zhipu':
            return ChatZhipuAI(
                    api_key=api,
                    model="glm-4-plus",
                    temperature=0.01,
                    streaming=False if callbacks is None else True,
                    callbacks=callbacks)
        if model == "deepseek":
            key = os.environ.get("DEEPSEEK_API_KEY")
            return ChatOpenAI(
                    model='deepseek-chat',
                    openai_api_base='https://api.deepseek.com',
                    openai_api_key=key,
                    max_tokens=4096,
                    streaming=False if callbacks is None else True,
                    callbacks=callbacks)
        elif model == 'qwen':
            model_id = 'Qwen/Qwen2.5-0.5B'
            model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
            llm = HuggingFacePipeline.from_model_id(
                model_id=model_id,
                task="text-generation",
                pipeline_kwargs={"max_new_tokens": 500})
            return ChatHuggingFace(
                        llm=llm,
                        streaming=False if callbacks is None else True,
                        callbacks=callbacks)
        raise ValueError("unknown model: %s" % model)


class TestAction(BaseAction):
    name = 'test'

    def run(self, conf):
        # llm = HuggingFaceEndpoint( repo_id="Qwen/Qwen2-0.5B")
        # llm = HuggingFacePipeline.from_model_id(model_id="Qwen/Qwen2-1.5B", task="text-generation")
        # llm = HuggingFacePipeline.from_model_id(
        #     model_id="Qwen/Qwen2-0.5B",
        #     task="text-generation", pipeline_kwargs={"max_new_tokens": 1024})
        # chat = ChatHuggingFace(llm=llm)
        class CallBack(BaseCallbackHandler):

            def on_llm_new_token(self, token: str, *args, **kwargs):
                print(token, end='', flush=True)
                super().on_llm_new_token(token, *args, **kwargs)

        chat = ChatFactory.get_chat('qwen', callbacks=[CallBack()])
        resp = chat.invoke("给我一个出游计划")
        print(resp)


class SearchAction(BaseAction):
    name = 'search'

    def add_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument('--data', type=str, required=True)
        parser.add_argument('question', type=str)
        EmbeddingFactor.add_parser(parser)

    def run(self, conf):
        db = Chroma(
            persist_directory=conf.data,
            embedding_function=EmbeddingFactor.get_embedding(conf))
        LOG.info(f"database info: {db._collection.count()}")

        docs = db.similarity_search(conf.question, k=3)
        for i, doc in enumerate(docs):
            print(f"检索到的第{i+1}个内容: \n{doc.page_content[:200]}", end="\n--------------\n")

        docs = db.max_marginal_relevance_search(conf.question, k=3)
        for i, doc in enumerate(docs):
            print(f"检索到的第{i+1}个内容: \n{doc.page_content[:200]}", end="\n--------------\n")


class GenIndexAction(BaseAction):
    name = 'gen_index'

    def add_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument('--data', type=str, required=True)
        parser.add_argument('--knowledge', type=str, required=True)
        parser.add_argument('--skip', type=int, default=0)
        parser.add_argument('--sleep' ,type=float, default=0.5)
        parser.add_argument('--max-docs', type=int, default=0)
        EmbeddingFactor.add_parser(parser)


    def run(self, conf):
        LOG.info(f"gen_index: {conf.data}, {conf.knowledge}")
        docs = list(split_documents(conf.knowledge, max_docs=conf.max_docs))
        split_docs = np.array_split(docs, len(docs)//1 + 1)
        split_count = len(split_docs)
        embedding = EmbeddingFactor.get_embedding(conf)
        for idx, d in enumerate(split_docs):
            if conf.skip and idx < conf.skip:
                continue
            if len(d) == 0:
                continue
            LOG.info("%d/%d Post docs: %d" % (idx, split_count, len(d)))
            try:
                db = Chroma.from_documents(
                    documents=list(d),
                    embedding=embedding,
                    client_settings=Settings(
                        anonymized_telemetry=False,
                        is_persistent=True,
                        persist_directory=conf.data),
                    persist_directory=conf.data)
            except:
                LOG.exception("Failed to index %d" % idx)
            finally:
                time.sleep(conf.sleep)


def split_documents(parent, max_docs=0):
    total = 0
    for root, _, files in os.walk(parent):
        for file in files:
            if file.find('SUMMARY') != -1:
                continue
            if file.endswith('.md'):
                fullpath = os.path.join(root, file)
                for doc in UnstructuredMarkdownLoader(fullpath).load():
                    if len(doc.page_content) < 100:
                        continue
                    head = fullpath[len(parent):].replace('/', ' ').replace('.md', '')
                    doc.page_content = f"# {head}\n" + doc.page_content
                    fixed_source = doc.metadata['source'][len(parent)+1:]
                    doc.metadata['source'] = fixed_source
                    yield doc
                    total += 1
                    if max_docs and total >= max_docs:
                        return


class SplitAction(BaseAction):
    name = 'split'

    def add_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument('--knowledge', type=str, required=True)

    def run(self, conf):
        for idx, doc in enumerate(split_documents(conf.knowledge)):
            print("%d %s" % (idx, '='*150))
            print(doc)
            print("=" * 150)


def get_all_subclasses(cls):
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--online', action='store_true')
    sub_parser = parser.add_subparsers(dest='subcommand', required=True)
    for action in [z() for z in get_all_subclasses(BaseAction)]:
        p = sub_parser.add_parser(action.name)
        action.add_parser(p)
        p.set_defaults(subcommand=action.run)

    conf = parser.parse_args()

    if conf.online:
        os.environ['TRANSFORMERS_OFFLINE'] = '0'
        os.environ['HF_DATASETS_OFFLINE'] = '0'
        os.environ['DISABLE_TELEMETRY'] = '0'
    else:
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['DISABLE_TELEMETRY'] = '1'

    level = logging.INFO
    if conf.debug:
        level = logging.DEBUG
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s.%(funcName)s:%(lineno)d %(message)s")
    conf.subcommand(conf)

if __name__ == '__main__':
    main()
