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

    def __init__(self):

        logger.info("Initializing LLM Client")

        if "GOOGLE_API_KEY" not in os.environ:
            raise RuntimeError("GOOGLE_API_KEY environment variable must be set.")

        logger.info("Google API key GOOGLE_API_KEY found")

        self.model_name = LLMConfig.model_name
        self.temperature = LLMConfig.temperature

        self.model = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature
        )

        logger.info("Model defined with model name: {}, temperature: {}".format(self.model_name, str(self.temperature)))

    def query_llm(self, origin, message):
        """
        Generic method for querying the LLM.

        :param origin: Method name, as string (used for logging)
        :param message: Message to send to the LLM, as list of LangChain messages
        :return: Content of the response, as string
        """

        # Invoke the LLM, keeping track of time
        start = time.time()
        response = self.model.invoke(message)
        duration = time.time() - start

        # Log the usage
        usage = getattr(response, "usage_metadata", {})
        self.log_llm_usage(origin, duration, message, response.content, usage)

        # Return the response content
        return response.content

    def describe_image(self, image_file_path, chunk_size):
        """
        Method to describe an image using the LLM. Requires a multi-modal LLM.

        :param image_file_path: Path to image file
        :param chunk_size: Desired chunk size, in characters
        :return: Textual description of the image, as string
        """

        logger.info("Describing image at: '{}'".format(image_file_path))

        # Define the purpose of the LLM, used as a system message
        purpose = """You are an expert in NVMe, NVMe-MI, PCIe, and other storage specifications.

                    Your task is to generate a strictly factual, literal description of the provided image 
                    so it can be stored in a vector database and later used to retrieve the same image. 
                    The description should focus on the content and meaning of the image, ad be grounded and 
                    technical.
                    
                    Follow these rules:
                    
                    1. ONLY describe what is visually present in the image.
                    2. DO NOT infer missing details, brand names, hardware models, device types, product 
                    families, or physical components unless they are explicitly shown.
                    3. If the image is a diagram, chart, table, or schematic, describe the layout, shapes, 
                    labels, and relationships between elements.
                    4. Do NOT assume the image shows real hardware unless real hardware is visually present.
                    5. If something is unclear or ambiguous, state that it is unclear rather than guessing.
                    6. If the image is mostly textual, read the text and provide a detailed technical summary as 
                    part of the description.
                    7. The description must be approximately {} characters.
                    
                    """.format(chunk_size)

        # Encode the image as base64
        encoded_image = self.encode_image(image_file_path)

        # Define the messages to send to the LLM
        messages = [
            SystemMessage(content=purpose),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Describe the following image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    }
                ]
            )
        ]

        # Query LLM and return response
        response = self.query_llm("describe_image", messages)
        return response

    def generate_questions(self, query):
        """
        Method to generate alternative search queries based on a given query. This is used for query expansion.

        :param query: Query, as string
        :return: List of alternative search queries, as list of strings
        """

        # Define the purpose of the LLM, used as a system message
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

        # Define the messages to send to the LLM
        messages = [
            SystemMessage(content=purpose),
            HumanMessage(
                content="Original question: {}\n\nGenerate helpful alternative search queries.".format(query)
            )
        ]

        # Query LLM, do some cleanup, and return response as list of strings
        response = self.query_llm("generate_questions", messages)
        response = response.replace("```json", "").replace("```", "").strip()
        return json.loads(response)

    def encode_image(self, image_path):
        """
        Method to encode an image as base64. This is required for sending images to the LLM.

        :param image_path: Path to image file
        :return: base64-encoded image, as string
        """

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        return base64.b64encode(image_bytes).decode("utf-8")

    def answer_with_context(self, query, relevant_chunks):

        logger.info(f"Answering question: '{query}', using {len(relevant_chunks)} refs.")

        # Define the purpose of the LLM, used as a system message
        purpose = """You are an expert in NVMe, NVMe-MI, PCIe, and other storage specifications.

                     Your task is to answer the provided question using content extracted from a specification.
                     You will be provided with a series of contexts consisting of text, tables (provided as images), drawings 
                     (provided as images), images (provided as images), and other media. Along with each context chunk
                     you will receive it's associated metadata, which includes spec origin, name, context type, and more.
                     After the context is provided, you will receive the query. When answering the query using the 
                     context provided, you must adhere to the following rules:
        
                     Rules:
                     1. Use ONLY the provided context.
                     2. For tables, use OCR for accurate interpretation.
                     3. For images that are mostly textual, read the text and use it as context in addition to the image.
                     4. If information is missing, say so.
                     5. You MUST include the relevant context for each portion of your response, surrounded by 
                        parenthesis, for example '(context 4,6)' 
    
                     CONTEXT:
                     
                     """


        human_message_content = []

        for i, chunk in enumerate(relevant_chunks):
            metadata = chunk.metadata
            content = chunk.page_content

            chunk_type = metadata.get("type", "text")

            metadata_string = "Type: {}, Source: {}, Page: {}".format(chunk_type, metadata["source"], metadata["page"])

            human_message_content.extend([
                {"type": "text", "text": "CONTEXT #{}".format(i+1)},
                {"type": "text", "text": "Metadata - {}".format(metadata_string)}
            ])

            if chunk_type == "text":

                human_message_content.extend([
                    {"type": "text", "text": "Content:"},
                    {"type": "text", "text": content}
                ])

            elif chunk_type == "table":
                image_path = metadata["table_path"]
                encoded_image = self.encode_image(image_path)

                human_message_content.extend([
                    {"type": "text", "text": "Content:"},
                    {"type": "image_url","image_url": {"url": "data:image/png;base64,{}".format(encoded_image)}}
                ])

            elif chunk_type == "drawing":
                image_path = metadata["drawing_path"]
                encoded_image = self.encode_image(image_path)

                human_message_content.extend([
                    {"type": "text", "text": "Content:"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,{}".format(encoded_image)}}
                ])

        human_message_content.append({"type": "text", "text": "\n\nQUERY: {}".format(query)})

        messages = [
            SystemMessage(content=purpose),
            HumanMessage(content=human_message_content)
        ]

        response = self.query_llm("answer_with_context", messages)

        return response, relevant_chunks

    def log_llm_usage(self, method, duration, prompt, response, usage = None):
        """
        Method to log LLM usage information

        :param method: Method name, as string
        :param duration: Duration of the LLM call, in seconds
        :param prompt: Prompt sent to the LLM, as string
        :param response: Response received from the LLM, as string
        :param usage: LLM usage metadata, as dict
        :return: None
        """

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