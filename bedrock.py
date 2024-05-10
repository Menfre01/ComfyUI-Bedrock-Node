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
                 "model": (["Claude3 sonnet", "Claude3 haiku"],),
                 "image": ("IMAGE",),
                 "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Generate prompt words based on the image, requiring a children's painting style and incorporate the phrase '(colorful, vibrant colors:1.2) (simple line art)' into the prompt. The result should only contain the stable diffusion prompt words without any other information."
                }),
            },
        }
        
    def IS_CHANGED(s):
        return time.time()
    
    def invoke(self, model, image, prompt)->str:
        tensor = image*255
        tensor = np.array(tensor, dtype=np.uint8)

        for i in range(tensor.shape[0]):
            image = Image.fromarray(tensor[i])
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        base64_data = base64.b64encode(image_bytes).decode("utf8")
        return self.cli.invoke_model(prompt, base64_data, "png", model)
    
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
    def _parse_model_id(self, model)->str:
        match model:
            case "Claude3 sonnet":
                return "anthropic.claude-3-sonnet-20240229-v1:0"
            case "Claude3 haiku":
                return "anthropic.claude-3-haiku-20240307-v1:0"
            case _:
                return ""
    def invoke_model(self, prompt, base64_image, media_type, model)->str:
        model_id = self._parse_model_id(model)
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": '''StableDiffusion is a deep learning text-to-image model that generates images based on prompts. These prompts can specify the desired elements of the image, such as the appearance of characters, background, color and lighting effects, as well as the theme and style of the image. The prompts often contain weighted numbers in parentheses to indicate the importance or emphasis of certain details. For example, "(masterpiece:1.5)" indicates that the quality of the work is very important. Multiple parentheses also have similar effects. In addition, if square brackets are used, such as "{blue hair:white hair:0.3}", this represents the fusion of blue and white hair, with blue hair accounting for 0.3.
Here is an example of using prompts to help an AI model generate an image: masterpiece,(bestquality),highlydetailed,ultra-detailed,cold,solo,(1girl),(detailedeyes),(shinegoldeneyes),(longliverhair),expressionless,(long sleeves),(puffy sleeves),(white wings),shinehalo,(heavymetal:1.2),(metaljewelry),cross-lacedfootwear (chain),(Whitedoves:1.2)	
Following the example, provide a set of prompts that detail the following content. Start the prompts directly without using natural language to describe them:''',
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
    model_id = "Claude3 sonnet"
    with open(image_path, "rb") as image_file:
        image = base64.b64encode(image_file.read()).decode("utf8")
    cli = BedrockCli()
    resp_prompt = cli.invoke_model(prompt_text, image, "jpeg", model_id)
    print(resp_prompt)