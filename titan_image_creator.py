import streamlit as st
import boto3
import json, base64, io
from random import randint
from PIL import Image

# Initialize the Bedrock runtime client
bedrock_runtime_client = boto3.client("bedrock-runtime")

# Define the titan_image function
def titan_image(
    payload: dict,
    num_image: int = 2,
    cfg: float = 10.0,
    seed: int = None,
    modelId: str = "amazon.titan-image-generator-v1",
) -> list:
    seed = seed if seed is not None else randint(0, 214783647)
    body = json.dumps(
        {
            **payload,
            "imageGenerationConfig": {
                "numberOfImages": num_image,
                "quality": "premium",
                "height": 1024,
                "width": 1024,
                "cfgScale": cfg,
                "seed": seed,
            },
        }
    )

    response = bedrock_runtime_client.invoke_model(
        body=body,
        modelId=modelId,
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    images = [
        Image.open(io.BytesIO(base64.b64decode(base64_image)))
        for base64_image in response_body.get("images")
    ]
    
    return images

# Define the Streamlit app
def main():
    st.title("Titan Image Generator")

    # Get user input
    text_input = st.text_input("Enter the text for the image generation:")

    if st.button("Generate Images"):
        # Generate images based on the input text
        if text_input:
            images = titan_image(
                {
                    "taskType": "TEXT_IMAGE",
                    "textToImageParams": {
                        "text": text_input,
                    },
                }
            )

            # Display the generated images
            for img in images:
                st.image(img, caption='Generated Image', use_column_width=True)
        else:
            st.warning("Please enter some text to generate images.")

if __name__ == "__main__":
    main()
