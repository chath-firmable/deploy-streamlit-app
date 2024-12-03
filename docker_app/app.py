import streamlit as st
import json
import boto3
from utils.auth import Auth
from utils.llm import Llm
from config_file import Config
# Basic page config
st.set_page_config(
    page_title="Your App Name",
    layout="wide"
)

# ID of Secrets Manager containing cognito parameters
secrets_manager_id = Config.SECRETS_MANAGER_ID

# ID of the AWS region in which Secrets Manager is deployed
region = Config.DEPLOYMENT_REGION

# Initialise CognitoAuthenticator
authenticator = Auth.get_authenticator(secrets_manager_id, region)

# Authenticate user, and stop here if not logged in
is_logged_in = authenticator.login()
if not is_logged_in:
    st.stop()


def logout():
    authenticator.logout()


with st.sidebar:
    st.text(f"Welcome,\n{authenticator.get_username()}")
    st.button("Logout", "logout_btn", on_click=logout)

# Add title on the page
st.title("Generative AI Application")

# Ask user for input text
input_sent = st.text_input("Input Sentence", "Say Hello World! in Spanish, French and Japanese.")
submit_button = st.button("Get LLM Response")


# Create the large language model object
try:
    bedrock_region = Config.BEDROCK_REGION
except AttributeError:
    bedrock_region = "ap-southeast-2"  # or whatever your default region should be
    
llm = Llm(bedrock_region)

# When there is an input text to process
if submit_button:
    # Invoke the Bedrock foundation model
    response = llm.invoke(input_sent)
    json_response = json.loads(response['body'].read().decode('utf-8'))
    st.write("**Foundation model output** \n\n", json_response['content'][0]['text'])


