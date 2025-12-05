import os
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily, ModelInfo
from dotenv import load_dotenv

load_dotenv()

def create_model_client(
    server: str = "deepseek",
    configs: dict = {}
):
    model_config = {
        "model": os.getenv("OPENAI_MODEL"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
    }
    model_config.update(configs)
    if server == "deepseek":
        model_info =  ModelInfo(
            family=ModelFamily.GPT_4O,
            vision=False,
            function_calling=True,
            json_output=True,
            structured_output=True,
            multiple_system_messages=True,  # Enable support for multiple system messages (needed for Memory)
        )
        model_config["model_info"] = model_info
    return OpenAIChatCompletionClient(**model_config)

__all__ = ["create_model_client"]
