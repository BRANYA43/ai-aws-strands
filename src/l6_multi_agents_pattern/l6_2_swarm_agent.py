import logging

from strands import Agent
from strands.multiagent import Swarm
from strands.models.litellm import LiteLLMModel

import config

model = LiteLLMModel(
    client_args={'api_key': config.API_KEY},
    model_id=config.MODEL_ID,
)

logging.getLogger('strands.multiagent').setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s | %(name)s | %(message)s', handlers=[logging.StreamHandler()])


researcher = Agent(
    name='researcher',
    model=model,
    system_prompt='You are a research specialist...',
)
coder = Agent(
    name='coder',
    model=model,
    system_prompt='You are a coder specialist...',
)
reviewer = Agent(
    name='reviewer',
    model=model,
    system_prompt='You are a reviewer specialist...'
)
architect = Agent(
    name='architect',
    model=model,
    system_prompt='You are an architect specialist...',
)

swarm = Swarm(
    nodes=[researcher, coder, architect],
    entry_point=researcher,
    max_handoffs=20,
    max_iterations=20,
    execution_timeout=900.0, # 15m
    node_timeout=300,  # 5m
    repetitive_handoff_detection_window=8,  # There must be >= 3 unique agents in the last 8 handoffs
    repetitive_handoff_min_unique_agents=3,
)

result = swarm('Design and implement a simple REST API for a todo app.')

print('Status:', result.status)
print('Node history:', [n.node_id for n in result.node_history])