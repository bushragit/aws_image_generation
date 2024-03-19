import streamlit as st
from PIL import Image
import base64
import boto3
import json
import io

# Initialize Boto3 client
bedrock_runtime_client = boto3.client("bedrock-runtime")

# Define the function to generate Titan images
def titan_image(payload: dict) -> list:
    modelId = "amazon.titan-image-generator-v1"
    response = bedrock_runtime_client.invoke_model(
        body=json.dumps(payload),
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


# Streamlit app
def main():
    st.title("Titan Image Generator")

    # Upload image
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    # Input text for background change
    text_input = st.text_input("Enter text for background change")

    # Mask prompt input
    mask_prompt_input = st.text_input("Enter mask prompt (optional)")

    # Generate button
    if st.button("Generate Image"):
        if uploaded_image is not None:
            # Convert uploaded image to base64
            image_bytes = uploaded_image.read()
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")

            # Define payload
            payload = {
                "taskType": "OUTPAINTING",
                "outPaintingParams": {
                    "text": text_input,
                    "image": encoded_image,
                    "maskPrompt": mask_prompt_input,
                    "outPaintingMode": "PRECISE",
                },
            }

            # Generate images
            generated_images = titan_image(payload)

            # Display input image
            st.subheader("Input Image")
            st.image(image_bytes, use_column_width=True, caption='Uploaded Image')

            # Display generated images
            st.subheader("Generated Images")
            for img in generated_images:
                st.image(img, use_column_width=True, caption='Generated Image')

        else:
            st.warning("Please upload an image first.")


if __name__ == "__main__":
    main()
