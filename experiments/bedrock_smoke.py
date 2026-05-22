from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(
    model_id="us.anthropic.claude-sonnet-4-6",
    region_name="us-east-1",
)

response = llm.invoke("Say hello.")
print("BEDROCK RESPONSE:", response.content)