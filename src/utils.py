from time import sleep

from retry import retry
from strands import Agent
from strands.agent import AgentResult


@retry(Exception, tries=5, delay=5)
def call_with_retry(agent: Agent, query, start_delay: int = None) -> AgentResult:
    if start_delay:
        sleep(start_delay)
    return agent(query)