import pytest
from ragas import SingleTurnSample, MultiTurnSample
from ragas.messages import HumanMessage, AIMessage
from ragas.metrics import Faithfulness, TopicAdherenceScore
from utils import load_test_data, get_llm_response


#@pytest.mark.parametrize("getData", load_test_data("Test4.json"), indirect=True)
@pytest.mark.asyncio
async def test_topicAdherence(llm_wrapper, getData):
    topicScore = TopicAdherenceScore(llm=llm_wrapper)
    score = await topicScore.multi_turn_ascore(getData)
    print(score)
    assert score > 0.8


@pytest.fixture
def getData():
    # test_data = request.param
    # responseDict = get_llm_response(test_data)
    conversation = [
        HumanMessage(content="how many articles are there in the selenium webdriver python course?"),
        AIMessage(content="There are 23 articles in the Selenium WebDriver Python course."),
        HumanMessage(content="How many downloadable resources are there in this course?"),
        AIMessage(content="There are 9 downloadable resources in the course.")

    ]
    reference = [""" 
    The AI should:
    1. Give results related to the selenium webdriver python course
    2. There are 23 articles and 9 downloadable resources in the course"""]
    sample = MultiTurnSample(user_input=conversation, reference_topics=reference)
    return sample
