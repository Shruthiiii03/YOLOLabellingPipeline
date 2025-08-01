import os
import base64
import mimetypes
from dotenv import load_dotenv

from PIL import Image
from io import BytesIO

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from image_utils import resize_img, plot_bounding_boxes

def config_client():
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    llm_client = ChatGoogleGenerativeAI(
        model = "gemini-2.0-flash",
        safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH},
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    return llm_client

#Utils - Parsing of json response
def parse_json(json_output: str):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output

def get_response(llm_client, im, user_req):
    mime_type = "image/jpeg"
    
    buffered = BytesIO()
    im.save(buffered, format="JPEG")
    buffered.seek(0)
    
    encoded_image = base64.b64encode(buffered.read()).decode("utf-8")
    
    user_message = HumanMessage(content=[
        {"type":"text", "text": user_req},
        {"type":"image_url", "image_url":f"data:{mime_type};base64,{encoded_image}"}
    ])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
                If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
                """
            ),
            user_message
        ]
    )

    chain = prompt | llm_client
    response = chain.invoke({})
    bounding_boxes = parse_json(response.content)

    return bounding_boxes

if __name__ == "__main__":
    #Example Pipeline Usage
    llm_client = config_client()
    img_path = "./sample_img/Cupcakes.png"
    user_req = "Detect the 2D bounding boxes of the cupcakes (with 'label' as topping description)"
    
    im = resize_img(img_path)

    bounding_boxes = get_response(llm_client, im, user_req)
    plot_bounding_boxes(im, bounding_boxes)