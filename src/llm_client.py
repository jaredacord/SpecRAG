import base64
import json
import logging
import os
import time

from google.api_core.exceptions import DeadlineExceeded

logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

class LLMClient:

    def __init__(self, config):
        """
        Initialize method for the LLMClient class.

        :param config: ConfigClient instance
        """

        self.config = config

        # Abort if GOOGLE_API_KEY is not set
        if "GOOGLE_API_KEY" not in os.environ:
            raise RuntimeError("GOOGLE_API_KEY environment variable must be set.")

        # Initialize LLM model
        self.model_name = self.config.llm_model_name
        self.temperature = self.config.llm_temperature
        self.model = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            timeout=self.config.timeout,
            max_retries=0
        )

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

        if encoded_image is None:
            return "Image Not Found"

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

        # Log response as well for traceability
        logger.info("Image at '{}' has the following description: '{}'"
                    "".format(image_file_path,
                              response.replace('\n', ' ').replace('\r', ' ')))

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

        # Attempt to read file, and return None if it fails
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

        except Exception as e:
            logger.error("Error reading image file '{}': {}".format(image_path, e))
            return None

        return base64.b64encode(image_bytes).decode("utf-8")

    def answer_with_context(self, query, relevant_chunks):
        """
        Method to answer a question using the provided context

        :param query: Query
        :param relevant_chunks: List of relevant chunks
        :return: Answer, as string
        """

        logger.info(f"Answering question: '{query}', using {len(relevant_chunks)} refs.")

        # Define the purpose of the LLM, used as a system message
        purpose = """You are an expert assistant for answering questions about NVMe, NVMe-MI, PCIe,
and related storage specifications.

Your task is to answer the provided question using ONLY the supplied context,
which is extracted from specification documents.

You will be provided with multiple context chunks. Each chunk may contain:
- Plain text
- Tables (provided as assets)
- Drawings or diagrams (provided as assets)
- Other assets or media

Each context chunk includes metadata such as specification name, source,
page number, and content type.

You MUST follow these rules strictly:

1. Use ONLY the provided context. Do NOT rely on prior knowledge.
2. For tables provided as assets:
   - Perform OCR and use ONLY the text that can be directly read from the image.
3. For assets or drawings that contain mostly text:
   - Read and use the visible text as contextual information.
4. Do NOT infer, extrapolate, reconcile, or assume information that is not
   explicitly present in the provided context.
5. If information required to answer part or all of the question is missing
   or underspecified in the context:
   - State this explicitly.
   - Provide a partial answer using only the information that IS specified.
6. If the question asks for structured output (e.g., tables, offsets, bit
   ranges, or field layouts):
   - Include ONLY fields and values that are explicitly defined in the context.
   - If a complete structure cannot be constructed, produce a PARTIAL structure.
   - Clearly mark missing values as:
     "Not specified in the provided context".
   - Do NOT attempt to reconcile definitions across multiple sections or pages.
7. If a context chunk’s content is unavailable or unreadable:
   - Skip ONLY that chunk and continue using the remaining context.
8. Prefer correctness and explicit uncertainty over completeness.
   A partial or limited answer is always acceptable.
9. If the context does not meaningfully support answering the question:
   - State this clearly and briefly, and explain why.

Citation requirements:
- You MUST include relevant citations for each factual statement.
- Citations must be enclosed in parentheses and reference the specification
  name and page number(s), for example:
  (NVMe-Base-2.0d, page 14) or (NVMe-Base-2.0d, pages 121–142).

Formatting guidance:
- Tables are allowed but must be minimal and strictly grounded in the context.
- Avoid speculative language.
- Do not add rows, columns, or values unless they are explicitly supported.

Your goal is to provide a grounded, bounded answer that reflects exactly what
the provided context supports—no more, no less.

CONTEXT:

                     
                     """


        human_message_content = []

        # Iterate through each context chunk given
        for i, chunk in enumerate(relevant_chunks):

            # Extract metadata and content
            metadata = chunk.metadata
            content = chunk.page_content

            # Get the context chunk type, and define the metadata string
            chunk_type = metadata.get("type", "text")
            metadata_string = "Type: {}, Source: {}, Page: {}".format(chunk_type, metadata["pdf_name"],
                                                                      metadata["page"])

            # Add the context chunk to the human message content
            human_message_content.extend([
                {"type": "text", "text": "CONTEXT from {}, page {}".format(metadata["pdf_name"],
                                                                           metadata["page"])},
                {"type": "text", "text": "Metadata - {}".format(metadata_string)}
            ])

            # If context chunk is text, add the text to the human message content
            if chunk_type == "text":

                human_message_content.extend([
                    {"type": "text", "text": "Content:"},
                    {"type": "text", "text": content}
                ])

            # If context chunk is a table, get the table image and add it to the human message content
            elif chunk_type == "table":

                # Get image path then image for the table
                image_path = metadata["table_path"]
                encoded_image = self.encode_image(image_path)

                # Skip if the image is unavailable
                if encoded_image is None:
                    human_message_content.extend([{"type": "text", "text": "Content Unavailable."}])
                    continue

                # Add the table image to the human message content
                human_message_content.extend([
                    {"type": "text", "text": "Content:"},
                    {"type": "image_url","image_url": {"url": "data:image/png;base64,{}".format(encoded_image)}}
                ])

            # If context chunk is a drawing, get the drawing image and add it to the human message content
            elif chunk_type == "drawing":

                # Get image path then image for the drawing
                image_path = metadata["drawing_path"]
                encoded_image = self.encode_image(image_path)

                # Skip if the image is unavailable
                if encoded_image is None:
                    human_message_content.extend([{"type": "text", "text": "Content Unavailable."}])
                    continue

                # Add the drawing image to the human message content
                human_message_content.extend([
                    {"type": "text", "text": "Content:"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,{}".format(encoded_image)}}
                ])

            # Otherwise, skip the context chunk
            else:
                human_message_content.extend([{"type": "text", "text": "Content Unavailable."}])

        # Now, add the query to the human message content
        human_message_content.append({"type": "text", "text": "\n\nQUERY: {}".format(query)})

        # Define the messages to send to the LLM
        messages = [
            SystemMessage(content=purpose),
            HumanMessage(content=human_message_content)
        ]

        # Query LLM
        response = self.query_llm("answer_with_context", messages)

        # Return the response and relevant chunks
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