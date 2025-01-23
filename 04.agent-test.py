#!/usr/bin/env python3
import logging

import os
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.globals import set_debug
from langchain.tools import BaseTool
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools import BingSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import BingSearchAPIWrapper
from langchain_google_community import GoogleSearchRun
from langchain_google_community import GoogleSearchAPIWrapper

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# 搜索工具
class SearchTool(BaseTool):
    name: str = "Search"
    description: str = "如果我想知道天气，请使用它"
    # return_direct = True  # 直接返回结果

    def _run(self, query: str) -> str:
        print("\nSearchTool query: " + query)
        return "你在北京： 气温： 10度， 湿度： 50%, 空气质量：优"


# 计算工具
class CalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = "如果是关于数学计算的问题，请使用它"
    return_direct: bool = True

    def _run(self, query: str) -> str:
        print("\nCalculatorTool query: " + query)
        return "100"


def main():
    set_debug(True)
    api = os.environ.get("ZHIPUAI_API_KEY")
    llm = ChatZhipuAI(api_key=api,
                      model="glm-4-plus",
                      temperature=0.1)
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
    google_api = GoogleSearchAPIWrapper()
    tools = [
             #WikipediaQueryRun(api_wrapper=api_wrapper),
             #GoogleSearchRun(api_wrapper=google_api),
             CalculatorTool()
    ]
    agent = initialize_agent(
                tools, llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True)

    while True:
        question = input("请输入问题:")
        r = agent.invoke(input=question)
        print("答案：%s" % r)


if __name__ == "__main__":
    main()
