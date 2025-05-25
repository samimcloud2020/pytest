import os
import pytest
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

os.environ[
    "OPENAI_API_KEY"] = ""


@pytest.fixture
def llm_wrapper():
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    lang_chain_llm = LangchainLLMWrapper(llm)
    return lang_chain_llm
