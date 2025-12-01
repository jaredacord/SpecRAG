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


    def answer_with_context(self, query, retrieved_docs):
        logger.info("Answering question: '{}', using {} references.".format(query, len(retrieved_docs)))

        start = time.time()
        context = "\n\n".join([d.page_content for d in retrieved_docs])
        prompt = f"""
    Use the following context from a PDF to answer the question.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {query}
    
    Answer the question using only the context.
    """

        response = self.model.invoke([HumanMessage(content=prompt)])

        duration = time.time() - start
        usage = getattr(response, "usage_metadata", {})

        self.log_llm_usage("answer_with_context", duration, prompt, response.content, usage)

        return {
            "answer": response.content,
            "chunks": retrieved_docs
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