from strands import Agent
from strands.models.litellm import LiteLLMModel
from strands_tools import http_request

import config

model = LiteLLMModel(
    client_args={'api_key': config.API_KEY},
    model_id=config.MODEL_ID,
)

researcher = Agent(
    name='researcher',
    model=model,
    callback_handler=None,
    tools=[http_request],
    system_prompt="""
    You are a researcher agent that gathers information form the web.
    1. Determine if the input is a research query or factual claim.
    2. Use your research tools (http_request) to find relevant information.
    3. Include source URLs and keep finding under 500 words.
    """
)
analyst = Agent(
    name='analyst',
    model=model,
    callback_handler=None,
    system_prompt="""
    You are an analyst agent that verifies information.
    1. For factual claims: Rate accuracy from 1-5 and correct if needed.
    2. For research queries: indentify 3-5 key insights.
    3. Evaluate source reliability and keep analysis under 400 words.
    """
)
writer = Agent(
    name='writer',
    model=model,
    system_prompt="""
    You are a writer agent that creates clear reports.
    1. For fact-checks: State whether claims are true or false.
    2. For research: Present key insights in a logical structure.
    3. Keep reports under 500 words with brief source mentions.
    """
)


def run_research_workflow(query: str):
    print('Researching:', query)
    research_response = researcher(f'Research: "{query}". Use tools to find reliable sources with URLs.')
    research_findings = str(research_response)
    print(f'Research completed ({len(research_findings)} chars)')

    analyst_response = analyst(f'Analyze these findings about "{query}":\n\n{research_findings}')
    analysis = str(analyst_response)
    print(f'Analysis completed ({len(analysis)} chars)')

    writer_response = writer(f'Create a report on "{query}" based on:\n\n{analysis}')
    report = str(writer_response)
    print(f'Report completed ({len(report)} chars)')

    return {
        'query': query,
        'research': research_findings,
        'analysis': analysis,
        'report': report
    }


def run_fact_check(claim: str):
    print('Fact-checking:', claim)

    research_response = researcher(f'Fact-check: "{claim}". Find evidence for/against this claim with sources.')
    research_findings = str(research_response)

    analyst_response = analyst(
        f'Analyze evidence for: "{claim}"\n\nResearch: {research_findings}\n\n'
        'Provide verdict (TRUE/FALSE/PARTIALLY TRUE), confidence level, and evidence.'
    )
    analysis = str(analyst_response)

    report_response = writer(f'Create fact-check report for: "{claim}"\n\nAnalysis: {analysis}')
    report = str(report_response)

    return {
        'claim': claim,
        'research': research_findings,
        'analysis': analysis,
        'report': report,
    }


def main():
    print('Multi-Agent Research Workflow Demo')
    print('=' * 50)

    query = 'Latest developments in AI safety'
    print('Query:', query)
    results = run_research_workflow(query)
    print('Final Report:')
    print('-' * 30)
    print(results['report'])

    print('\n' + '=' * 50)
    claim = "OpenAI's GPT-3 was released in March 2023"
    print(f'\n Claim:', claim)
    fact_results = run_fact_check(claim)
    print('\n Fact-Check Report:')
    print('-' * 30)
    print(fact_results['report'])


if __name__ == '__main__':
    main()

