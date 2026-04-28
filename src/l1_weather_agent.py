from strands import Agent
from strands.models.litellm import LiteLLMModel
from strands_tools import http_request

import config
from utils import call_with_retry

WEATHER_SYSTEM_PROMPT = """
You are a friendly and helpful weather assistant with HTTP capabilities.

Your primary function is to provide accurate weather forecasts for locations in the Ukraine by using 
the National Weather Service API.

Follow these steps to perform a user's request:
1.  First, if you don't have grid coordinates, use the points API endpoint to get them.
    - For latitude and longitude: https://api.weather.gov/points/{latitude},{longitude}
    - For a US zipcode: https://api.weather.gov/points/{zipcode}
2.  The points API will return a `forecast` URL. Use this URL to make a second HTTP request to get the actual weather forecast.
3.  Process the forecast data and present it to the user in a clear, easy-to-understand format.

When displaying your response:
-   Highlight key information like temperature, precipitation, and any weather alerts.
-   Explain technical terms in simple language.
-   If you get an error, apologize and explain that you couldn't retrieve the weather information.
"""

def create_weather_agent() -> Agent:
    # Configure LLM.
    model = LiteLLMModel(
        client_args={'api_key': config.API_KEY},
        model_id=config.MODEL_ID,

        params={'max_tokens': 1500, 'temperature': 0.7}
    )

    # Create the agent instance
    agent = Agent(
        system_prompt=WEATHER_SYSTEM_PROMPT,
        tools=[http_request],
        model=model,
    )
    return agent


def main():
    weather_agent = create_weather_agent()
    user_query = 'Compare the temperature in New York, NY and Chicago, this tomorrow.'

    response = call_with_retry(weather_agent, user_query)

    print('User query:\n', user_query)
    print('Agent response:\n', response)

if __name__ == '__main__':
    main()