from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.models.litellm import LiteLLMModel
from strands.tools.mcp import MCPClient

import config
from utils import call_with_retry

SYSTEM_PROMPT = """
You are an expert on Amazon Web Services.
Use the provided tools to answer questions about AWS services 
based on the official documentation. Always provide accurate, 
up-to-date information from the AWS docs.
"""


model = LiteLLMModel(
    client_args={'api_key': config.API_KEY},
    model_id=config.MODEL_ID,
)

mcp_client = MCPClient(
    transport_callable=lambda: stdio_client(
        StdioServerParameters(
            command='uvx', args=['awslabs.aws-documentation-mcp-server@latest'],
        )
    )
)

with mcp_client:
    aws_tools = mcp_client.list_tools_sync()
    print('Successfully loaded', len(aws_tools), 'tools from the MCP server.')

    agent = Agent(
        model=model,
        tools=aws_tools,
        system_prompt=SYSTEM_PROMPT,
    )

    query = 'What is the maximum invocation payload size from AWS Lambda.'
    print('--- Querying AWS Documentation ---')
    print('User Query:', query, '\n')

    response = call_with_retry(agent, query)

    print('--- Agent Response ---')
    print(response)