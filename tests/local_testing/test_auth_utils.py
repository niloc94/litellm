# What is this?
## Tests if proxy/auth/auth_utils.py works as expected

import sys, os, asyncio, time, random, uuid
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm.proxy.auth.auth_utils import (
    _allow_model_level_clientside_configurable_parameters,
)
from litellm.router import Router


@pytest.mark.parametrize(
    "allowed_param, input_value, should_return_true",
    [
        ("api_base", {"api_base": "http://dummy.com"}, True),
        (
            {"api_base": "https://api.openai.com/v1"},
            {"api_base": "https://api.openai.com/v1"},
            True,
        ),  # should return True
        (
            {"api_base": "https://api.openai.com/v1"},
            {"api_base": "https://api.anthropic.com/v1"},
            False,
        ),  # should return False
        (
            {"api_base": "^https://litellm.*direct\.fireworks\.ai/v1$"},
            {"api_base": "https://litellm-dev.direct.fireworks.ai/v1"},
            True,
        ),
        (
            {"api_base": "^https://litellm.*novice\.fireworks\.ai/v1$"},
            {"api_base": "https://litellm-dev.direct.fireworks.ai/v1"},
            False,
        ),
    ],
)
def test_configurable_clientside_parameters(
    allowed_param, input_value, should_return_true
):
    router = Router(
        model_list=[
            {
                "model_name": "dummy-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "dummy-key",
                    "configurable_clientside_auth_params": [allowed_param],
                },
            }
        ]
    )
    resp = _allow_model_level_clientside_configurable_parameters(
        model="dummy-model",
        param="api_base",
        request_body_value=input_value["api_base"],
        llm_router=router,
    )
    print(resp)
    assert resp == should_return_true


def test_get_end_user_id_from_request_body_always_returns_str():
    from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body

    request_body = {"user": 123}
    end_user_id = get_end_user_id_from_request_body(request_body)
    assert end_user_id == "123"
    assert isinstance(end_user_id, str)


@pytest.mark.parametrize(
    "request_data, route, expected_model",
    [        
        # default cases, take from request header
        ({"target_model_names": "gpt-3.5-turbo, gpt-4o-mini-general-deployment"}, "/v1/files", ["gpt-3.5-turbo", "gpt-4o-mini-general-deployment"]),
        ({"target_model_names": "gpt-3.5-turbo"}, {}, ["gpt-3.5-turbo"]),
        ({"model": "gpt-3.5-turbo, gpt-4o-mini-general-deployment"}, {}, ["gpt-3.5-turbo", "gpt-4o-mini-general-deployment"]),
        ({"model": "gpt-3.5-turbo"}, {}, "gpt-3.5-turbo"),
        ({"model": "gpt-3.5-turbo, gpt-4o-mini-general-deployment"}, {}, ["gpt-3.5-turbo", "gpt-4o-mini-general-deployment"]),

        # Azure OpenAI passthrough - take from route
        ({}, "/openai/deployments/gpt-4-deployment/chat/completions", "gpt-4-deployment"),
    
        # Bedrock passthrough
        ({}, "/bedrock/model/anthropic.claude-3-5-sonnet-20240620-v1:0/converse", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
        ({"model": "gpt-4"}, "/bedrock/model/anthropic.claude-3-5-sonnet-20240620-v1:0/converse", "anthropic.claude-3-5-sonnet-20240620-v1:0"),  # Request body ignored
    ],
)
def test_get_model_from_request(request_data, route, expected_model):
    from litellm.proxy.auth.auth_utils import get_model_from_request

    model = get_model_from_request(request_data, route)
    assert model == expected_model

@pytest.mark.parametrize(
    "input_route, expected_model",
    [
        # Valid Bedrock routes
        ("/bedrock/model/anthropic.claude-3-5-sonnet-20240620-v1:0/converse", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
        ("/bedrock/model/cohere.command-text-v14/converse-stream", "cohere.command-text-v14"),
        ("/bedrock/model/arn:aws:bedrock:us-west-2:123456789012:custom-model/my-model-abc/invoke", "arn:aws:bedrock:us-west-2:123456789012:custom-model/my-model-abc"),

        # Invalid cases
        ("/bedrock/model/anthropic.claude/invalid-command", None),
        ("/bedrock/model/", None),
        ("/bedrock/model/some-model", None),  # Missing command
        ("/chat/completions", None),  # Non-bedrock route
    ],
)
def test_get_model_from_bedrock_runtime_requests(input_route, expected_model):
    from litellm.proxy.auth.auth_utils import _get_model_from_bedrock_runtime_requests

    model = _get_model_from_bedrock_runtime_requests(input_route)
    assert model == expected_model
