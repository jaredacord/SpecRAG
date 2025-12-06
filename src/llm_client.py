import base64
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import LLMConfig


class LLMClient:

    def __init__(self, model_name = LLMConfig.model_name, temperature = LLMConfig.temperature):

        logger.info("Initializing LLM Client")

        if "GOOGLE_API_KEY" not in os.environ:
            raise RuntimeError("GOOGLE_API_KEY environment variable must be set.")

        logger.info("Google API key GOOGLE_API_KEY found")

        self.model_name = model_name
        self.temperature = temperature

        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature
        )

        logger.info("Model defined with model name: {}, temperature: {}".format(model_name, str(temperature)))

    def summarize_image(self, image_file_path, chunk_size):
        prompt = """You are an expert in the field of NVMe flash storage devices. Given an image, produce a detailed, 
        factual, and technical description of the image so that this text can later be stored in a vector 
        database and used to retrieve the original image. The description must be around {} characters.
        Image:""".format(chunk_size)

        logger.info("Summerizing image at: '{}'".format(image_file_path))

        start = time.time()
        encoded_image = self.encode_image(image_file_path)

        message_local = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{encoded_image}"},
            ]
        )
        response = self.model.invoke([message_local])
        duration = time.time() - start
        usage = getattr(response, "usage_metadata", {})

        self.log_llm_usage("summarize_image", duration, prompt, response.content, usage)

        return response.content

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def answer_with_context(self, query, relevent_chunks):
        logger.info(f"Answering question: '{query}', using {len(relevent_chunks)} refs.")

        start = time.time()
        message_content = []

        context_contents = []

        for i, chunk in enumerate(relevent_chunks):
            metadata = chunk.metadata
            content = chunk.page_content

            metadata_as_text = json.dumps(metadata)
            chunk_type = metadata.get("type", "text")

            context_content = ""

            if chunk_type == "text":

                context_content += "Relevant Chunk Number: {} \n".format(i)
                context_content += "Metadata: {} \n".format(metadata_as_text)
                context_content += "Content: {} \n\n".format(content)

                context_contents.append({"type": "text", "text": context_content})

            elif chunk_type == "table":
                image_path = metadata.get("table_path", "")
                encoded_image = self.encode_image(image_path) if image_path else ""

                context_content += "Relevant Chunk Number: {} \n".format(i)
                context_content += "Metadata: {} \n".format(metadata_as_text)
                context_content += "Content: \n"

                context_contents.append({"type": "text", "text": context_content})
                context_contents.append({"type": "image_url", "image_url": f"data:image/png;base64,{encoded_image}"},)

        temp_promt_1 = f"""
            You are answering a question using content extracted from a PDF. 
            The context may include text or tables (provided as images).
        
            Rules:
            1. Use ONLY the provided context.
            2. For tables, use OCR for accurate interpretation.
            3. If information is missing, say so.
        
            CONTEXT:
            """
        temp_promt_2 = f"""
            QUESTION:
            {query}
    
            Answer using only the provided context.
            """

        message_content.append({"type": "text", "text": temp_promt_1})
        message_content.extend(context_contents)
        message_content.append({"type": "text", "text": temp_promt_2})

        response = self.model.invoke([HumanMessage(content=message_content)])

        duration = time.time() - start
        usage = getattr(response, "usage_metadata", {})

        return {
            "answer": response.content,
            "chunks": relevent_chunks
        }

    def log_llm_usage(self, method, duration, prompt, response, usage = None):

        record = {
            "method": method,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model_name,
            "duration_seconds": round(duration, 3),
            "prompt_length": len(prompt),
            "response_length": len(response),
            "usage": usage or {}
        }

        log_str = "[LLM Usage]" + json.dumps(record)

        logger.info(log_str)