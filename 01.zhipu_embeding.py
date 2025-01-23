#!/usr/bin/env python3

import os

import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import UnstructuredRSTLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from zhipuai import ZhipuAI

db_path = './data/chroma'

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_embedding(text, api):
    c = ZhipuAI(api_key=api)
    resp = c.embeddings.create(input=text, model='embedding-3')
    return resp.data[0].embedding

def main():
    api = os.environ.get("ZHIPUAI_API_KEY")
    texts = [
        '王子'*100,
        '皇子',
    ]

    t1 = get_embedding(texts[0], api)
    t2 = get_embedding(texts[1], api)
    print(cos_sim(t1, t2))


    # loader = UnstructuredRSTLoader('/home/jeffrey/99cloud/ict-iaas/ict-doc/doc/operator/faq.rst')
    # t = loader.load()
    # t3 = None
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # split_docs = text_splitter.split_documents(t)
    # vectordb = Chroma.from_documents(
    #         documents=split_docs,
    #         embedding=ZhipuAIEmbeddings(api_key=api),
    #         persist_directory=db_path)
    # vectordb.persist()

    # print(vectordb._collection.count())

    # question = 'ovs-dpdk 修改为普通ovs'

    # sim_docs = vectordb.similarity_search(question, k=3)
    # for i, sim_doc in enumerate(sim_docs):
    #     print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")



if __name__ == "__main__":
    main()
