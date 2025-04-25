"""
Model configuration file for tests.

Note: The appropriate environment variables need to be set up for each provider/model tested.
"""

# ==== FOR TESTING DIFFERENT MODEL PROVIDERS ====

# Different model providers and corresponding models to be tested
# Modify as needed to test different models
# Note that this list of models is only used for basic testing
PROVIDERS_TO_TEST = {
    'openai': 'gpt-4o-mini',
    'azureopenai': 'gpt-4o-mini',
    'anthropic': 'claude-3-5-sonnet-latest',
    'gemini': 'gemini-2.0-flash',
    'deepseek': 'deepseek-chat',
    'openrouter': 'google/gemma-3-4b-it:free',
}

# ==== FOR TESTING ALL AGENTS, TOOLS AND STRUCTURED OUTPUTS ====

SEARCH_PROVIDER = 'serper'

# Note that the models need to support tool use

REASONING_MODEL_PROVIDER = 'openai'
REASONING_MODEL = 'gpt-4o-mini'

MAIN_MODEL_PROVIDER = 'openai'
MAIN_MODEL = 'gpt-4o-mini'

FAST_MODEL_PROVIDER = 'openai'
FAST_MODEL = 'gpt-4o-mini'