import asyncio
import atexit
import logging
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from typing import Any, Coroutine
from langchain_core.messages import HumanMessage, AIMessage


class Chatbot:
    """Handle the data and provide functionalities of the chatbot."""

    _SUPPORTED_PROVIDERS = {
        "openai",
        "google_genai",
    }

    _SUPPORTED_MODELS_BY_PROVIDER = {
        "openai": {
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
        },
        "google_genai": {
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
        },
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
        self._chat_history = []
        self._event_loop = None
        atexit.register(self._cleanup_event_loop)

    def _get_or_create_event_loop(self) -> asyncio.AbstractEventLoop:
        """Setup the event loop for asynchronous operations.
        
        Returns:
            asyncio.AbstractEventLoop: The event loop.
        Raises:
            Exception: If unable to create or get an event loop.
        """
        try:
            event_loop = asyncio.get_event_loop()
            if event_loop.is_closed():
                event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(event_loop)
        except RuntimeError:
            event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(event_loop)
        except Exception as e:
            error_msg = f"Error setting up event loop: {e}"
            self._logger.error(error_msg)
            raise e
        return event_loop
    
    def _cleanup_event_loop(self) -> None:
        """Cleanup the event loop and resources.
        
        Ensures that any pending tasks are cancelled and the event loop is
        properly closed when the application exits.

        Using atexit to register this cleanup function is more reliable than
        calling it on __del__ method, as __del__ may not be called in some
        situations (e.g., if there are circular references).
        """
        if self._event_loop and not self._event_loop.is_closed():
            try:
                pending = asyncio.all_tasks(self._event_loop)
                for task in pending:
                    task.cancel()
            except RuntimeError:
                # No tasks to cancel
                pass
            self._event_loop.close()

    def _run_async(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        """Run asynchronous coroutine using managed event loop.
        
        Args:
            coroutine (Coroutine): The coroutine to run.
        Returns:
            Any: The result of the coroutine.
        Raises:
            TypeError: If the provided argument is not a coroutine.
            Exception: If the coroutine execution fails.
            asyncio.CancelledError: If the coroutine is cancelled.
        """
        if not isinstance(coroutine, Coroutine):
            error_msg = (
                f"Invalid coroutine type. Got {type(coroutine).__name__}")
            self._logger.error(error_msg)
            raise TypeError(error_msg)
        if self._event_loop is None or self._event_loop.is_closed():
            self._event_loop = self._get_or_create_event_loop()
        if self._event_loop.is_running():
            task = asyncio.create_task(coroutine)
            return task
        else:
            try:
                return self._event_loop.run_until_complete(coroutine)
            except asyncio.CancelledError as e:
                warning_msg = f"Coroutine was cancelled: {e}"
                self._logger.warning(warning_msg)
                raise e                
            except Exception as e:
                error_msg = f"Unexpected error in async execution: {e}"
                self._logger.error(error_msg)
                raise e

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
        
    def _validate_model_provider(self, model_provider: str) -> None:
        """Validate the model provider.
        
        Args:
            model_provider (str): The model provider.
        Raises:
            TypeError: If model_provider is not a string.
            ValueError: If model_provider is not supported or empty.
        """
        self._validate_string(model_provider, "Model provider")
        if model_provider not in self._SUPPORTED_PROVIDERS:
            error_msg = (f"Model provider '{model_provider}' is not "
                f"supported.")
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

    def _init_model(self) -> BaseChatModel:
        """Initialize the chat model.
        
        Returns:
            BaseChatModel: The initialized chat model.
        Raises:
            ValueError: If model provider, model name, or API key is not set.
            Exception: If the model initialization fails.
        """
        if self._model_provider is None:
            error_msg = "Model provider is not set."
            self._logger.error(error_msg)
            raise ValueError(error_msg)
        if self._model_name is None:
            error_msg = "Model name is not set."
            self._logger.error(error_msg)
            raise ValueError(error_msg)
        if self._api_key.get(self._model_provider, None) is None:
            error_msg = (f"API key for model provider '{self._model_provider}' "
                f"is not set.")
            self._logger.error(error_msg)
            raise ValueError(error_msg)
        try:
            return init_chat_model(
                f"{self._model_provider}:{self._model_name}",
                temperature=self._model_temperature,
                api_key=self._api_key[self._model_provider],
            )
        except Exception as e:
            error_msg = f"Failed to initialize chat model: {e}"
            self._logger.error(error_msg)
            raise e
        
    def _create_agent(self) -> CompiledStateGraph:
        """Create the agent.
        
        Returns:
            CompiledStateGraph: The created agent.
        Raises:
            ValueError: If model provider, model name, or API key is not set.
            Exception: If the model initialization or agent creation fails.
        """
        if self._model is None:
            self._model = self._init_model()
        try:
            return create_react_agent(
                model=self._model,
                tools=[],
                checkpointer=self._checkpointer,
                prompt=self._system_message,
            )
        except Exception as e:
            error_msg = f"Failed to create agent: {e}"
            self._logger.error(error_msg)
            raise e
        
    async def _async_get_response(
            self,
            user_message: HumanMessage
        ) -> AIMessage | None:
        """Asynchronously get a response from the chatbot.

        Args:
            user_message (HumanMessage): The user's message.
        Returns:
            AIMessage | None: The agent's response message, or None if an
                error occurred.
        Raises:
            Exception: If there is an error getting the response from the agent.
        """
        if self._agent is None:
            self._agent = self._create_agent()
        try:
            response = await self._agent.ainvoke(
                {"messages": [user_message]},
                {"configurable": {"thread_id": "1"}},
            )
            self._chat_history.append(user_message)
            agent_message = response["messages"][-1]
            self._chat_history.append(agent_message)
            return agent_message
        except Exception as e:
            self._logger.error(f"Error getting response: {e}")
            raise e

    def set_model_provider(self, model_provider: str | None) -> None:
        """Set the model provider.
        
        Args:
            model_provider (str | None): The model provider to set. If None,
                the model provider is unset.
        Raises:
            TypeError: If model_provider is not a string or None.
            ValueError: If model_provider is not supported or empty.
        """
        if model_provider is not None:
            self._validate_model_provider(model_provider)
        if model_provider != self._model_provider:
            self._model_provider = model_provider
            # Reset model name when provider changes to avoid invalid
            # combinations of provider and model name.
            self._model_name = None
            self._reset_model_and_agent()

    def get_model_provider(self) -> str | None:
        """Get the model provider."""
        return self._model_provider

    def set_model_name(self, model_name: str | None) -> None:
        """Set the model name.
        
        Args:
            model_name (str | None): The model name to set. If None, the model
                name is unset.
        Raises:
            TypeError: If model_name is not a string or None.
            ValueError: If model_name is not supported, empty, or if the model
                provider is not set (in case the model name is not None).
        """
        if model_name is not None:
            if self._model_provider is None:
                error_msg = ("Model provider must be set before setting the "
                    "model name.")
                self._logger.error(error_msg)
                raise ValueError(error_msg)
            self._validate_string(model_name, "Model name")
            supported_models = self._SUPPORTED_MODELS_BY_PROVIDER.get(
                self._model_provider, 
                {},
            )
            if model_name not in supported_models:
                error_msg = f"Model '{model_name}' is not supported."
                self._logger.error(error_msg)
                raise ValueError(error_msg)
        if model_name != self._model_name:
            self._model_name = model_name
            self._reset_model_and_agent()

    def get_model_name(self) -> str | None:
        """Get the model name."""
        return self._model_name

    def set_model_temperature(self, model_temperature: float) -> None:
        """Set the model temperature.
        
        Args:
            model_temperature (float): The model temperature to set. Must be
                in the range [0.0, 1.0].
        Raises:
            TypeError: If model_temperature is not a number.
            ValueError: If model_temperature is out of range.
        """
        if not isinstance(model_temperature, (int, float)):
            error_msg = (f"Model temperature must be a number, got "
                f"{type(model_temperature).__name__}")
            self._logger.error(error_msg)
            raise TypeError(error_msg)
        model_temperature = float(model_temperature)
        if not (0.0 <= model_temperature <= 1.0):
            error_msg = (f"Model temperature {model_temperature} is out of "
                f"range [0.0, 1.0]")
            self._logger.error(error_msg)
            raise ValueError(error_msg)
        if model_temperature != self._model_temperature:
            self._model_temperature = model_temperature
            self._reset_model_and_agent()

    def get_model_temperature(self) -> float:
        """Get the model temperature."""
        return self._model_temperature
    
    def set_system_message(self, system_message: str) -> None:
        """Set the system message.
        
        Args:
            system_message (str): The system message to set.
        Raises:
            TypeError: If system_message is not a string.
            ValueError: If system_message is empty.
        """
        self._validate_string(system_message, "System message")
        if system_message != self._system_message:
            self._system_message = system_message
            self._reset_model_and_agent()

    def get_system_message(self) -> str:
        """Get the system message."""
        return self._system_message

    def set_api_key(self, model_provider: str, api_key: str) -> None:
        """Set the API key for a model provider.
        
        Args:
            model_provider (str): The model provider.
            api_key (str): The API key to set.
        Raises:
            TypeError: If model_provider or api_key is not a string.
            ValueError: If model_provider is not supported or empty, or if
                api_key is empty.
        """
        self._validate_model_provider(model_provider)
        self._validate_string(api_key, "API key")
        if api_key != self._api_key.get(model_provider, None):
            self._api_key[model_provider] = api_key
            self._reset_model_and_agent()

    def get_api_key(self, model_provider: str) -> str | None:
        """Get the API key for a model provider."""
        return self._api_key.get(model_provider, None)
        
    def get_response(self, user_message: HumanMessage) -> AIMessage | None:
        """Get a response from the chatbot.
        
        Args:
            user_message (HumanMessage): The user's message.
        Returns:
            AIMessage | None: The agent's response message, or None if an
                error occurred.
        Raises:
            TypeError: If user_message is not a HumanMessage.
            Exception: If there is an error getting the response from the agent.
        """
        if not isinstance(user_message, HumanMessage):
            error_msg = (f"user_message must be a HumanMessage, got "
                f"{type(user_message).__name__}")
            self._logger.error(error_msg)
            raise TypeError(error_msg)
        return self._run_async(self._async_get_response(user_message))
    
    def get_chat_history(self) -> list[HumanMessage | AIMessage]:
        """Return the chat history.

        Returns:
            list[HumanMessage | AIMessage]: The chat history.
        """
        return self._chat_history.copy()
    