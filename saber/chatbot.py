import logging
from langgraph.checkpoint.memory import InMemorySaver


class Chatbot:
    """Handle the data and provide functionalities of the chatbot."""

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model_provider = None
        self._model_name = None
        self._model_temperature = 0.0
        self._system_message = (f"You are an assistant for question-answering "
            f"tasks. Use the following pieces of retrieved context to answer "
            f"the question. If you don't know the answer, just say that you "
            f"don't know. Use three sentences maximum and keep the answer "
            f"concise.")
        self._api_key = {}
        self._checkpointer = InMemorySaver()
        self._model = None
        self._agent = None
        self._agent_config = {"configurable": {"thread_id": "1"}}
        self._chat_history = []
