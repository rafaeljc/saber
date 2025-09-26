import logging
from langgraph.checkpoint.memory import InMemorySaver


class Chatbot:
    """Handle the data and provide functionalities of the chatbot."""

    _SUPPORTED_PROVIDERS = {
        "openai",
        "google_genai",
    }

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

    def _validate_string(self, value: str, var_name: str) -> None:
        """Validate that a variable is a non-empty string.
        
        Args:
            value (str): The value to validate.
            var_name (str): The name of the variable (for error messages).
        Raises:
            TypeError: If the value is not a string.
            ValueError: If the string is empty.
        """
        if not isinstance(value, str):
            error_msg = (f"{var_name} must be a string, got "
                f"{type(value).__name__}")
            self._logger.error(error_msg)
            raise TypeError(error_msg)
        if value == "":
            error_msg = f"{var_name} must be a non-empty string."
            self._logger.error(error_msg)
            raise ValueError(error_msg)

    def _reset_model_and_agent(self) -> None:
        """Reset the model and agent to their initial state.
        
        Must be called when any parameter affecting them is changed. This
        ensures that they will be re-initialized with the new parameters only
        when next used.
        """
        self._model = None
        self._agent = None