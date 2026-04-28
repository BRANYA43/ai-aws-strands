from mcp import stdio_client, StdioServerParameters
from strands import Agent
from strands.models.litellm import LiteLLMModel
from strands.tools.mcp import MCPClient

import config
from utils import call_with_retry

SYSTEM_PROMPT = """
You are a helpful travel assistant with access to both web search 
and accommodation search capabilities. Use the appropriate tools 
to help users find information and plan their travels.
"""

model = LiteLLMModel(
    client_args={'api_key': config.API_KEY},
    model_id=config.MODEL_ID,
    params={
        'stream': False,
        'max_tokens': 1500,
    }
)

web_search_mcp_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command='npx',
            args=['-y', 'mcp-remote', 'https://mcp.exa.ai/mcp'],
            env={'EXA_API_KEY': config.EXA_API_KEY},  # type: ignore
        )
    )
)

airbnb_mcp_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command='npx',
            args=['-y', '@openbnb/mcp-server-airbnb', '--ignore-robots-txt'],
        )
    )
)

with web_search_mcp_client, airbnb_mcp_client:
    web_search_tools = web_search_mcp_client.list_tools_sync()
    airbnb_tools = airbnb_mcp_client.list_tools_sync()
    all_tools = web_search_tools + airbnb_tools

    print('Loaded', len(web_search_tools), 'web search tools.')
    print('Loaded', len(airbnb_tools), 'Airbnb tools.')
    print('Total tools available:', len(all_tools))

    agent = Agent(
        tools=all_tools,
        model=model,
        system_prompt=SYSTEM_PROMPT,
    )

    # Query the agent
    query = 'What is the fastest way to get to Barcelona from London?'
    print('--- Querying with Multiple MCP Servers ---')
    print('User Query:', query)

    response = call_with_retry(agent, query)

    print('--- Agent Response ---')
    print(response)

    query2 = 'Find rooms in Barcelona for 2 people for 2 nights?'
    print('--- Querying with Multiple MCP Servers ---')
    print('User Query:', query2)

    response = call_with_retry(agent, query2)

    print('--- Agent Response ---')
    print(response)
