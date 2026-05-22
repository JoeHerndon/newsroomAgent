import os
from langchain_anthropic import ChatAnthropic

ANTHROPIC_MODEL = "claude-sonnet-4-6"
BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"
OLLAMA_MODEL = "llama3.2:3b"


def get_chat_model(temperature: float = 0):
    """RETURN A CHAT MODEL BASED ON MODEL_PROVIDER ENV VAR."""
    provider = os.environ.get("MODEL_PROVIDER", "anthropic")

    if provider == "anthropic":
        return ChatAnthropic(model=ANTHROPIC_MODEL, temperature=temperature)

    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse
        return ChatBedrockConverse(
            model_id=BEDROCK_MODEL,
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            temperature=temperature,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, temperature=temperature)

    raise ValueError(f"Unknown MODEL_PROVIDER: {provider}")