from pydantic import BaseModel, Field
from strands import Agent
from strands.models.litellm import LiteLLMModel

import config
from utils import call_with_retry

SYSTEM_PROMPT = """
Your are an expert assistant 
that extracts structured information about people from text 
based on the provided schema.
"""

TEXT_TO_PROCESS = """
John Smith is a 30-year-old software engineer living in San Francisco.
"""


class PersonInfo(BaseModel):
    name: str = Field(..., description='The full name of the person.')
    age: int = Field(..., description='The age of the person.')
    occupation: str = Field(..., description='The current occupation of the person')
    address: str = Field(..., description='The place of living.')


def main():
    model = LiteLLMModel(
        client_args={'api_key': config.API_KEY},
        model_id=config.MODEL_ID,
    )

    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        structured_output_model=PersonInfo,
    )

    print('--- Extracting information from text ---\n')
    print('Input text:', TEXT_TO_PROCESS, '\n')

    try:
        response = call_with_retry(agent, TEXT_TO_PROCESS)
        person_info: PersonInfo = response.structured_output
        print('--- Extraction Successful ---')
        print('Name:', person_info.name)
        print('Age:', person_info.age)
        print('Occupation:', person_info.occupation)
        print('Address:', person_info.address)

    except Exception as e:
        print('--- Extraction failed ---')
        print('An error occurred:', e)


if __name__ == '__main__':
    main()
