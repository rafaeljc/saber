"""This module implements the settings page for the application, providing a
comprehensive interface for configuring chatbot parameters and model settings.
It handles real-time configuration updates with validation and error handling.

Configuration Categories:
    **Model Settings:**
    - Provider selection
    - Model name selection (provider-specific)
    - API key input with secure password field

    **Parameter Settings:**
    - Temperature control (0.0 to 1.0 range)
    - System message customization

Architecture:
    - **Cached Resources**: Provider and model lists cached for performance
    - **Callback System**: Real-time updates using Streamlit's on_change events

User Experience Flow:
    1. User selects provider from dropdown
    2. Model list updates dynamically based on provider
    3. User selects model from updated list
    4. User enters API key for selected provider
    5. User adjusts temperature and system message
    6. All changes are applied immediately with validation
    7. Errors are displayed with helpful guidance

Performance Optimizations:
    - Cached provider and model lists
    - Lazy loading of configuration options
    - Efficient state management

Note:
    This module requires the chatbot to be properly initialized in the session
    state. All configuration changes are applied immediately and persist
    throughout the session.
"""

import streamlit as st
from typing import Callable


# Retrieve chatbot instance from Streamlit session state
# This is initialized by the main application (app.py) when the session starts
chatbot = st.session_state.get("chatbot", None)


@st.cache_resource
def get_provider_list() -> list[str]:
    """Retrieves and caches the available LLM providers that the chatbot
    supports.

    Returns:
        list[str]: a list of supported provider names.

    Note:
        Uses Streamlit's @st.cache_resource decorator to cache the result
        across the entire application session. The cache is cleared when
        the application restarts.
    """
    return list(chatbot.get_supported_providers())


@st.cache_resource
def get_model_list_by_provider(model_provider: str) -> list[str]:
    """Retrieves and caches the available model names for a given provider.

    Args:
        model_provider (str): The provider name. Must be one of the supported
            providers from get_provider_list().

    Returns:
        list[str]: a list of supported model names for the provider.

    Note:
        Uses Streamlit's @st.cache_resource decorator to cache the result
        across the entire application session. The cache is cleared when
        the application restarts.
    """
    return list(chatbot.get_supported_models_by_provider(model_provider))


@st.cache_resource
def get_index(item_list: list[str], item: str) -> int | None:
    """Utility function that helps set the initial selected value in Streamlit
    select widgets by finding the index of the current value in the options
    list.

    Args:
        item_list (list[str]): The list of options/items to search through.
        item (str): The item to locate within the list. This is usually
            the current value that should be pre-selected in the widget.

    Returns:
        int | None: The zero-based index of the item if found, or None if
            the item is not in the list. Streamlit uses None to indicate
            no initial selection.

    Note:
        Cached using @st.cache_resource to avoid repeated list searches,
        especially useful for longer option lists.
    """
    try:
        return item_list.index(item)
    except ValueError:
        return None


@st.cache_resource
def get_set_functions_dict() -> dict[str, Callable]:
    """Create a mapping of chatbot attributes to their setter functions.

    This function provides a centralized mapping between configuration
    attribute names and their corresponding chatbot setter methods. It
    enables the callback system to dynamically invoke the appropriate
    setter based on which UI widget was changed.

    Returns:
        dict[str, Callable]: A dictionary mapping attribute names to their
            corresponding chatbot setter functions:
            - "model_provider": chatbot.set_model_provider
            - "model_name": chatbot.set_model_name
            - "api_key": chatbot.set_api_key
            - "model_temperature": chatbot.set_model_temperature
            - "system_message": chatbot.set_system_message

    Note:
        The function mapping is cached to avoid recreating the dictionary
        on every callback invocation, improving performance for frequent
        UI updates.
    """
    return {
        "model_provider": chatbot.set_model_provider,
        "model_name": chatbot.set_model_name,
        "api_key": chatbot.set_api_key,
        "model_temperature": chatbot.set_model_temperature,
        "system_message": chatbot.set_system_message,
    }


def set_value(chatbot_attr: str) -> None:
    """Callback function to update chatbot configuration in real-time.

    This is the central callback function that handles all configuration
    updates from the settings UI. It retrieves values from Streamlit's
    session state and applies them to the chatbot using the appropriate
    setter functions.

    Args:
        chatbot_attr (str): The name of the chatbot attribute to update.
            Must be one of: "model_provider", "model_name", "api_key",
            "model_temperature", "system_message".

    Special Case:
        **API Key Handling:**
        - Requires the current provider to be set first
        - Constructs session state key as "api_key_{provider}"

    Note:
        This function expects Streamlit session state to contain the
        updated values when called. It's designed to work with Streamlit's
        widget callback system.
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
            st.error("No API key value found in session state")
            return
        args = {"model_provider": current_provider, "api_key": api_key_value}
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
    """Creates a comprehensive model configuration interface with three columns:
    provider selection, model selection, and API key input.

    Layout:
        **Column 1 - Provider Selection:**
        - Dropdown with all supported providers
        - Pre-selected with current provider
        - Triggers model list update on change

        **Column 2 - Model Selection:**
        - Dynamic dropdown based on selected provider
        - Shows provider-specific models only
        - Pre-selected with current model

        **Column 3 - API Key Input:**
        - Secure password field for API key
        - Disabled until provider is selected
        - Pre-filled with current API key (masked)
    """
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
    """Creates an interface for adjusting advanced model parameters that
    control the behavior and output characteristics of the AI responses.

    Layout:
        **Model Temperature (Slider):**
        - Range: 0.0 to 1.0 with 0.05 precision

        **System Message (Text Area):**
        - Multi-line text input for AI behavior instructions
        - Defines the AI's role, personality, and response style
        - 200px height for comfortable editing.
    """
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
    """Creates the main settings page that orchestrates all configuration
    sections and provides a comprehensive interface for chatbot setup.
    """
    st.title("Settings")
    display_model_settings()
    display_parameters_settings()


# Main execution: Initialize settings page or show error
if chatbot is not None:
    settings_page()
else:
    st.error("Chatbot is not initialized.")
