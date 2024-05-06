import os, logging, boto3, json, base64, time, numpy as np
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class Bedrock:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "invoke"
    OUTPUT_NODE = False
    CATEGORY = "tag/bedrock"
    def __init__(self):
        self.cli = BedrockCli()
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{ 
                 "model_id": (["anthropic.claude-3-sonnet-20240229-v1:0"],),
                 "image": ("IMAGE",),
                 "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Generate prompt words based on the image, requiring a children's painting style and incorporate the phrase '(colorful, vibrant colors:1.2) (simple line art)' into the prompt. The result should only contain the stable diffusion prompt words without any other information."
                }),
            },
        }
        
    def IS_CHANGED(s):
        return time.time()
    
    def invoke(self, model_id, image, prompt)->str:
        tensor = image*255
        tensor = np.array(tensor, dtype=np.uint8)

        for i in range(tensor.shape[0]):
            image = Image.fromarray(tensor[i])
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        base64_data = base64.b64encode(image_bytes).decode("utf8")
        return self.cli.invoke_model(prompt, base64_data, "png", model_id)
    
class BedrockCli:
    def __init__(self, client=None):
        if client != None:
          self.client = client
        else:
          load_dotenv()
          ak = os.getenv('AWS_ACCESS_KEY_ID')
          sk = os.getenv('AWS_SECRET_ACCESS_KEY')
          if ak == None or sk == None:
              logging.error('Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env file')
              raise ValueError('Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env file')
          self.client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1', aws_access_key_id=ak, aws_secret_access_key=sk)
    def invoke_model(self, prompt, base64_image, media_type, model_id)->str:
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/" + media_type,
                                "data": base64_image,
                            },
                        },
                    ],
                }
            ],
        }
        try:
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body),
            )
            result = json.loads(response.get("body").read())
            output_list = result.get("content", [])
            output_prompt = output_list[0]["text"]
            logger.info("prompt: %s", output_prompt)
            return output_prompt
        except ClientError as err:
            logger.error(
                "Couldn't invoke Claude 3 Sonnet. Here's why: %s: %s",
                err.response["Error"]["Code"],
                err.response["Error"]["Message"],
            )
            raise

if __name__ == "__main__":
    image_path = "images/snow.jpeg"
    prompt_text = "Generate prompt words based on the image, requiring a children's painting style and incorporate the phrase '(colorful, vibrant colors:1.2) (simple line art)' into the prompt. The result should only contain the stable diffusion prompt words without any other information."
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    with open(image_path, "rb") as image_file:
        image = base64.b64encode(image_file.read()).decode("utf8")
    cli = BedrockCli()
    resp_prompt = cli.invoke_model(prompt_text, image, "jpeg", model_id)
    print(resp_prompt)