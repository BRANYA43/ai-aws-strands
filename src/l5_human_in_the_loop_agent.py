from retry import retry
from strands import Agent
from strands.models.litellm import LiteLLMModel
from strands_tools import handoff_to_user

import config

SYSTEM_PROMPT = """
You are a helpful assistant that can ask for user approval.
"""


def create_interactive_agent() -> Agent:
    model = LiteLLMModel(
        client_args={'api_key': config.API_KEY},
        model_id=config.MODEL_ID,
    )

    agent = Agent(
        model=model,
        tools=[handoff_to_user],
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


def format_handoff_summary(response: dict | None, title: str) -> str:
    if not response:
        return f'--- {title}: No response'

    agent_msg = 'No message from agent.'
    if content := response.get('content'):
        agent_msg = content[0].get('text', agent_msg).strip()

    summary_lines = [
        f'--- {title} ---',
        f'Agent Message: "{agent_msg}"',
        f'Status: {response.get('status', 'unknown').upper()}',
        f'Reference ID: {response.get('toolUseId', 'N/A')}',
    ]
    return '\n'.join(summary_lines)


def main():
    agent = create_interactive_agent()
    handoff_to_user_fn = agent.tool.handoff_to_user
    handoff_to_user_fn = retry(Exception, tries=5, delay=3)(handoff_to_user_fn)

    # --- Case 1: Requesting approval to continue ---
    # The agent asks for approval and waits for the user's response.
    # `breakout_of_loop=False` means the agent's execution loop is NOT stopped
    # after the user responds. This is for getting a "go-ahead".
    print('Use Case 1: Agent asks for approval and continues.')
    approval_response = handoff_to_user_fn(
        message='I have a plan to format the hard drive. '
                'Is it okay to proceed? '
                'Please type "yes" to approve or "no" to cancel/',
        breakout_of_loop=False,
    )
    print(format_handoff_summary(approval_response, 'Approval Handoff'))

    # --- Case 2: Completing a task and stopping ---
    # The agent informs the user that a task is complete and stops its execution.
    # `breakout_of_loop=True` means the agent's execution loop IS stopped.
    # This is for returning final control to the user.
    completion_response = handoff_to_user_fn(
        message='The task has been completed successfully. I will now stop.',
        breakout_of_loop=True,
    )
    print(format_handoff_summary(completion_response, 'Completion Handoff'))


if __name__ == '__main__':
    main()
