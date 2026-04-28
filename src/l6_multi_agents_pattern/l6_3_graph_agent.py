import logging
from strands import Agent
from strands.multiagent import GraphBuilder
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
    system_prompt='You are a research specialist focused on gathering comprehensive data and information',
)
analyst = Agent(
    name='analyst',
    model=model,
    system_prompt='You are a analyst specialist who processes and interprets research data.'
)
fact_checker = Agent(
    name='fact_checker',
    model=model,
    system_prompt='You are a fact checking specialist who validates information accuracy.'
)
report_writer = Agent(
    name='report_writer',
    model=model,
    system_prompt='You are a report writing specialist who creates structured, comprehensive reports.'
)

builder = GraphBuilder()

builder.add_node(researcher, 'research')
builder.add_node(analyst, 'analyst')
builder.add_node(fact_checker, 'fact_checker')
builder.add_node(report_writer, 'report_writer')

builder.add_edge('research', 'analyst')
builder.add_edge('research', 'fact_checker')
builder.add_edge('analyst', 'report_writer')
builder.add_edge('fact_checker', 'report_writer')

builder.set_entry_point('research')

builder.set_execution_timeout(600)  # 10m

graph = builder.build()

query = 'Research the impact of AI on healthcare and create a comprehensive report.'

result = graph(query)

print()
print('Status:', result.status)
print('Execution order', [n.node_id for n in result.execution_order])