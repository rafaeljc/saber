import streamlit as st
from typing import Callable


chatbot = st.session_state.get("chatbot", None)


@st.cache_resource
def get_provider_list() -> list[str]:
    """Get the list of supported model providers.
    
    Returns:
        list[str]: The list of supported providers.
    """
    return list(chatbot.get_supported_providers())


@st.cache_resource
def get_model_list_by_provider(model_provider: str) -> list[str]:
    """Get the list of supported models for a given provider.
    
    Args:
        model_provider (str): The provider name.
    Returns:
        list[str]: The list of supported models.
    """
    return list(chatbot.get_supported_models_by_provider(model_provider))


@st.cache_resource
def get_index(item_list: list[str], item: str) -> int | None:
    """Get the index of an item in a list.
    
    Args:
        item_list (list[str]): The list to search.
        item (str): The item to find.
    Returns:
        int | None: The index of the item or None if not found.
    """
    try:
        return item_list.index(item)
    except ValueError:
        return None


@st.cache_resource
def get_set_functions_dict() -> dict[str, Callable]:
    """Get the set functions for Chatbot attributes that can be modified in the 
    settings.
    
    Returns:
        dict[str, Callable]: A dictionary mapping attribute names to their
            corresponding set functions.
    """
    return {
        "model_provider": chatbot.set_model_provider,
        "model_name": chatbot.set_model_name,
        "api_key": chatbot.set_api_key,
        "model_temperature": chatbot.set_model_temperature,
        "system_message": chatbot.set_system_message,
    }


def set_value(chatbot_attr: str) -> None:
    """Callback function to set a Chatbot attribute.

    Args:
        chatbot_attr (str): The name of the attribute to set.
    """
    args = {}
    if chatbot_attr == "api_key":
        current_provider = st.session_state.get("model_provider", None)
        if current_provider is None:
            st.error("No provider selected for API key")
            return
        key = f"api_key_{current_provider}"
        api_key_value = st.session_state.get(key, None)
        if api_key_value is None:
            st.error(f"No API key value found in session state")
            return
        args = {
            "model_provider": current_provider,
            "api_key": api_key_value
        }
    else:
        value = st.session_state.get(chatbot_attr, None)
        if value is None:
            st.error(f"No value found in session state for: {chatbot_attr}")
            return
        args[chatbot_attr] = value
    set_functions = get_set_functions_dict()
    if chatbot_attr in set_functions:
        try:
            set_functions[chatbot_attr](**args)
        except Exception as e:
            st.error(f"Error setting {chatbot_attr}: {e}")
    else:
        st.error(f"No set function found for chatbot attribute: {chatbot_attr}")


def display_model_settings() -> None:
    """Display model settings options."""
    st.subheader("Model")
    model_provider = chatbot.get_model_provider()
    model_name = chatbot.get_model_name()
    col1, col2, col3 = st.columns(3)
    with col1:
        providers = get_provider_list()
        _ = st.selectbox(
            "Provider",
            providers,
            index=get_index(providers, model_provider),
            placeholder="Choose a provider",
            on_change=set_value,
            args=("model_provider",),
            key="model_provider",
        )
    with col2:
        models = get_model_list_by_provider(model_provider)
        _ = st.selectbox(
            "Model",
            models,
            index=get_index(models, model_name),
            placeholder="Choose a model",
            on_change=set_value,
            args=("model_name",),
            key="model_name",
        )
    with col3:
        disabled = model_provider is None
        api_key_value = chatbot.get_api_key(model_provider) or ""
        _ = st.text_input(
            "API Key",
            type="password",
            value=api_key_value,
            placeholder="Enter API key",
            on_change=set_value,
            args=("api_key",),
            key=f"api_key_{model_provider}",
            disabled=disabled,
        )


def display_parameters_settings() -> None:
    """Display model parameters settings options."""
    st.subheader("Parameters")
    _ = st.slider(
        "Model Temperature",
        min_value=0.0,
        max_value=1.0,
        value=chatbot.get_model_temperature(),
        step=0.05,
        on_change=set_value,
        args=("model_temperature",),
        key="model_temperature",
    )
    _ = st.text_area(
        "System Message",
        value=chatbot.get_system_message(),
        placeholder="Enter system message",
        height=200,
        on_change=set_value,
        args=("system_message",),
        key="system_message",
    )


def settings_page() -> None:
    """Render the settings page."""
    st.title("Settings")
    display_model_settings()
    display_parameters_settings()


if chatbot is not None:
    settings_page()
else:
    st.error("Chatbot is not initialized.")
