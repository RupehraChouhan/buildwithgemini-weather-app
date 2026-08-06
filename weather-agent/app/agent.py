# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from zoneinfo import ZoneInfo
import uuid

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.tools import ToolContext
from google.genai import types, Client
from google.genai.types import GenerateContentConfig, Modality
from google.cloud import storage


MODEL = "gemini-3.6-flash"
BUCKET_NAME = "qwiklabs-gcp-04-906b917dde28-static-assets-bucket"


def get_weather(query: str) -> str:
    """Simulates a web search. Use it get information on weather.

    Args:
        query: A string containing the location to get weather information for.

    Returns:
        A string with the simulated weather information for the queried location.
    """
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 15.5°C and foggy."
    return "It's 32.2°C and sunny."


def get_current_time(query: str) -> str:
    """Simulates getting the current time for a city.

    Args:
        city: The name of the city to get the current time for.

    Returns:
        A string with the current time information.
    """
    if "sf" in query.lower() or "san francisco" in query.lower():
        tz_identifier = "America/Los_Angeles"
    else:
        return f"Sorry, I don't have timezone information for query: {query}."

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    return f"The current time for query {query} is {now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"


async def generate_weather_image(location_and_weather: str, tool_context: ToolContext) -> dict:
    """Generates an image of the weather for a requested location and uploads it to public Cloud Storage.

    Args:
        location_and_weather: A description of the weather and location to generate an image for (e.g., 'foggy weather in San Francisco').

    Returns:
        A dictionary containing the public HTTPS URL of the generated image and status.
    """
    client = Client(vertexai=True, location='global')
    
    prompt = f"A beautiful high-quality photograph showing the weather: {location_and_weather}"
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-image",
        contents=prompt,
        config=GenerateContentConfig(
            response_modalities=[Modality.TEXT, Modality.IMAGE],
        ),
    )
    
    image_bytes = None
    mime_type = "image/jpeg"
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            image_bytes = part.inline_data.data
            mime_type = part.inline_data.mime_type or "image/jpeg"
            break
            
    if not image_bytes:
        return {"error": "Failed to generate image. No image data returned from model."}
        
    ext = "jpg" if "jpeg" in mime_type else "png"
    image_name = f"weather_{uuid.uuid4().hex[:8]}.{ext}"
    part = types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_bytes))
    await tool_context.save_artifact(image_name, part)
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(image_name)
    blob.upload_from_string(image_bytes, content_type=mime_type)
    
    public_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{image_name}"
    
    return {
        "status": "success",
        "image_url": public_url,
        "artifact_name": image_name
    }


root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction="You are a helpful AI assistant designed to provide accurate and useful information. Always present temperature values in Celsius. Whenever responding to any weather request or weather question, you MUST call both the get_weather tool to get the details AND the generate_weather_image tool to create a visual representation of that weather, and present BOTH the text details and the generated weather image URL to the user.",
    tools=[get_weather, get_current_time, generate_weather_image],
)

app = App(
    root_agent=root_agent,
    name="app",
)
