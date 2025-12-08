import base64
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage
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

    def query_llm(self, origin, message):

        start = time.time()
        response = self.model.invoke(message)
        duration = time.time() - start

        usage = getattr(response, "usage_metadata", {})
        self.log_llm_usage(origin, duration, message, response.content, usage)

        return response.content

    def describe_image(self, image_file_path, chunk_size):

        logger.info("Describing image at: '{}'".format(image_file_path))

        purpose = """You are an expert in NVMe flash storage devices and NVMe specifications.

                    Your task is to generate a strictly factual, literal description of the provided image 
                    so it can be stored in a vector database and later used to retrieve the same image.
                    
                    Follow these rules:
                    
                    1. ONLY describe what is visually present in the image.
                    2. DO NOT infer missing details, brand names, hardware models, device types, product 
                    families, or physical components unless they are explicitly shown.
                    3. If the image is a diagram, chart, table, or schematic, describe the layout, shapes, 
                    labels, colors, and relationships between elements.
                    4. Do NOT assume the image shows real hardware unless real hardware is visually present.
                    5. If something is unclear or ambiguous, state that it is unclear rather than guessing.
                    6. The description must be approximately {} characters.
                    
                    Begin with: "This image shows..." and continue with a grounded, technical description.
                    """.format(chunk_size)

        with open(image_file_path, "rb") as f:
            image_bytes = f.read()

        messages = [
            SystemMessage(content=purpose),
            HumanMessage(
                content="Image: ",
                image=image_bytes
            )
        ]

        response = self.query_llm("describe_image", messages)

        return response

    def generate_questions(self, query):

        purpose = """You are an expert in NVMe, NVMe-MI, PCIe, and storage specifications. 
                    Your task is to generate 3 to 5 alternative search queries that can be used
                    to retrieve relevant chunks from a vector database.
                    The expansions must remain faithful to the user's intent.\n\n
                    Guidelines:\n
                    - Produce 3 to 5 alternative search queries.\n
                    - Use terminology and aliases as they appear in NVMe/PCIe specs.\n
                    - No assumptions or invented details.\n
                    - DO NOT answer the question.\n
                    - Output ONLY a JSON array of strings, e.g.: ["query1", "query2"]."""

        messages = [
            SystemMessage(content=purpose),
            HumanMessage(
                content="Original question: {}\n\nGenerate helpful alternative search queries.".format(query)
            )
        ]

        response = self.query_llm("generate_questions", messages)

        response = response.replace("```json", "").replace("```", "").strip()

        return json.loads(response)

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