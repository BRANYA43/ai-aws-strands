from strands import Agent
from strands.models.litellm import LiteLLMModel
from strands.session.file_session_manager import FileSessionManager

import config
from utils import call_with_retry

SYSTEM_PROMPT = 'You are a friendly assistant. Keep your responses concise'


def create_persistent_agent(session_id: str) -> Agent:
    model = LiteLLMModel(
        client_args={'api_key': config.API_KEY},
        model_id=config.MODEL_ID,
    )

    session_manager = FileSessionManager(
        session_id=session_id,
        storage_dir=config.SESSION_STORAGE_DIR,
    )

    agent = Agent(
        model=model,
        session_manager=session_manager,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


def main():
    session_id = 'user_stalker_123'
    agent = create_persistent_agent(session_id)

    print('--- Conversation Start ---')

    print()

    query = 'Hey, my name is Bogdan.'
    print('User:', query)
    response1 = call_with_retry(agent, query)
    print('Agent:', response1)

    print()

    query = 'Do you remember my name?'
    print('User:', query)
    response2 = call_with_retry(agent, query)
    print('Agent', response2)

    print()

    print('--- Conversation End ---')


if __name__ == '__main__':
    main()