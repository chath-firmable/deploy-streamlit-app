import boto3
import json


class Llm:
    def __init__(self, bedrock_region):
        # Create Bedrock client
        bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=bedrock_region,
        )
        self.bedrock_client = bedrock_client

    def invoke(self, input_text):
        """
        Make a call to the foundation model through Bedrock using the Messages API
        """
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {
                    "role": "user",
                    "content": input_text
                }
            ],
            "max_tokens": 4096,
            "temperature": 0
        }
        body = json.dumps(body)
        accept = 'application/json'
        contentType = 'application/json'

        # Make the API call to Bedrock
        response = self.bedrock_client.invoke_model(
            body=body, 
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            accept=accept, 
            contentType=contentType
        )

        return response
