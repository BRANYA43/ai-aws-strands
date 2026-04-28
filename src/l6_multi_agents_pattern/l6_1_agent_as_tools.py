from strands import Agent, tool
from strands.models.litellm import LiteLLMModel
from strands_tools import http_request, retrieve

import config
from utils import call_with_retry


model_for_tool = LiteLLMModel(
    client_args={'api_key': config.API_KEY},
    model_id=config.MODEL_ID,
)

@tool
def research_assistant(query: str) -> str:
    """
    A specialized agent for research-related queries.

    This agent uses the `retrieve` and `http_request` tools to find factual well-sourced information.

    Args:
        query: A research question requiring factual information.

    Returns:
        A detailed research answer, ideally with citations.
    """
    print('--- Delegation to Research Assistant ---')
    agent = Agent(
        model=model_for_tool,
        system_prompt="""
        You are a specialized research assistant.
        Your sole purpose is to provide factual, well-sourced information.
        Always cite your sources when possible.
        """,
        tools=[retrieve, http_request],
    )

    response = call_with_retry(agent, query, start_delay=5)
    return str(response)


@tool
def product_recommendation_assistant(query: str) -> str:
    """
    A specialized agent for handling product recommendation queries.

    This agent can search for products and provide personalized recommendations.

    Args:
        query: A product inquiry with user preferences.

    Returns:
        Personalized product recommendations with clear reasoning.
    """
    print('--- Delegation to Product Recommendation Assistant ---')
    agent = Agent(
        model=model_for_tool,
        system_prompt="""
        You are a specialized product recommendation assistant.
        Provide personalized product suggestions based on user preferences.
        """,
        tools=[retrieve, http_request],
    )
    response = call_with_retry(agent, query, start_delay=5)
    return str(response)


@tool
def trip_planning_assistant(query: str) -> str:
    """
    A specialized agent for creating travel itineraries and giving travel advice.

    Args:
        query: A travel planning request with destination and preferences.

    Returns:
        A detailed travel itinerary or relevant travel advice.
    """
    print('--- Delegating to Trip Planning Assistant ---')
    agent = Agent(
        model=model_for_tool,
        system_prompt="""
        Your are a specialized travel planning assistant.
        Create detailed travel itineraries based on user preferences, including 
        recommendations for flights, accommodations and activities.
        """,
        tools=[retrieve, http_request],
    )
    response = call_with_retry(agent, query, start_delay=5)
    return str(response)


def create_orchestrator_agent() -> Agent:
    model = LiteLLMModel(
        client_args={'api_key': config.API_KEY},
        model_id=config.MODEL_ID,
    )

    agent = Agent(
        model=model,
        system_prompt="""
        You are a master assistant that routes complex queries to a team of specialized agents. 
        Based on the user's request, determine the best tool to use
        - For research questions and factual information -> Use the `research_assistant`.
        - For product recommendations and shopping -> Use the `product_recommendation_assistant`.
        - For travel planning and itineraries -> Use the `trip_planning_assistant`.
        - For simple greetings or questions your can answer directly -> Use Answer without using a tool.
        
        If a query requires multiple steps (e.g. planing a trip AND recommending products for it),
        call the necessary assistants in a logical sequence.
        """,
        tools=[
            research_assistant,
            product_recommendation_assistant,
            trip_planning_assistant,
        ],
    )
    return agent


def main():
    orchestrator = create_orchestrator_agent()

    user_query = 'I am planning a hiking trip to Patagonia next month and need recommendations for waterproof boots.'

    print('--- Orchestrator Agent ---')
    print('User Query:', user_query)

    final_response = call_with_retry(orchestrator, user_query)

    print('--- Final Response from Orchestrator ---')
    print(final_response)


if __name__ == '__main__':
    main()