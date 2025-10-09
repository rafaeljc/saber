"""This module contains the test suite for the Chatbot class, ensuring robust
validation of all configuration management and core functionality.

Testing Patterns:
    **Fixtures:**
    - Consistent test setup with predefined configurations
    - Reusable chatbot instances for different test scenarios
    - Isolation between test cases
    
    **Validation Strategy:**
    - Positive testing: Valid inputs produce expected results
    - Negative testing: Invalid inputs raise appropriate exceptions
    - Edge cases: Boundary conditions and special values
    - State preservation: Changes don't affect unrelated attributes

Usage:
    Run all tests:
        $ pytest tests/test_chatbot.py
    
    Run specific test class:
        $ pytest tests/test_chatbot.py::TestModelProviderAttributeManagement
    
    Run with coverage:
        $ pytest tests/test_chatbot.py --cov=saber.chatbot

Note:
    These tests ensure the chatbot behaves correctly under all conditions
    and provides reliable error handling. They serve as both validation
    and documentation of expected behavior.
"""

import pytest
from langchain_core.messages import HumanMessage

from saber.chatbot import Chatbot


class TestModelProviderAttributeManagement:
    """This test class validates the complete lifecycle of model provider
    management, including selection, validation, error handling, and
    state transitions.
    
    Test Coverage:
        - **Type Validation**: Ensures only string or None types accepted
        - **Provider Validation**: Verifies only supported providers allowed
        - **State Management**: Confirms proper attribute updates and resets
        - **Error Handling**: Validates appropriate exception types and messages
        - **Dependency Management**: Tests provider-model relationship handling
        
    Critical Behaviors Tested:
        - Invalid providers rejected with ValueError
        - Invalid types rejected with TypeError
        - Empty strings rejected appropriately
        - Setting provider resets dependent attributes (model_name)
        - State remains consistent after errors
    """

    @pytest.fixture
    def chatbot(self):
        """Provides a chatbot instance in a partially configured state suitable
        for testing provider-related operations. The instance has a valid
        provider set but no model name, allowing tests to verify provider
        functionality and provider-model dependencies.
        
        Configuration:
            - Provider: "google_genai" (valid, supported provider)
            - Model: None (not set, ready for model selection tests)
            - API Key: None (not set, prevents actual API calls)
        
        Returns:
            Chatbot: Configured instance ready for provider testing
        """
        cb = Chatbot()
        cb.set_model_provider("google_genai")
        return cb
    
    @pytest.fixture
    def chatbot_with_model(self):
        """Provides a chatbot instance in a more complete configuration state,
        suitable for testing operations that require both provider and model
        to be set. This fixture is used to test dependency relationships
        and state transitions.
        
        Configuration:
            - Provider: "google_genai" (valid, supported provider)
            - Model: "gemini-2.5-flash" (valid model for the provider)
            - API Key: None (not set, prevents actual API calls)
        
        Returns:
            Chatbot: Fully configured instance ready for advanced testing
        """
        cb = Chatbot()
        cb.set_model_provider("google_genai")
        cb.set_model_name("gemini-2.5-flash")
        return cb

    def test_set_invalid_type(self, chatbot):
        """Validates that the chatbot properly validates input types and raises
        appropriate exceptions for non-string, non-None values. This ensures
        type safety and prevents configuration corruption from invalid inputs.
        
        Test Strategy:
            - Arrange: Get current provider value for comparison
            - Act: Attempt to set invalid type (integer)
            - Assert: TypeError raised, original value preserved
        
        Error Handling Verification:
            - Exception type: TypeError (not ValueError or generic Exception)
            - State preservation: Original value remains unchanged
            - Input validation: Non-string types properly rejected
        """
        prev_value = chatbot.get_model_provider()
        non_valid_type = 12345
        with pytest.raises(TypeError):
            chatbot.set_model_provider(non_valid_type)
        assert chatbot.get_model_provider() == prev_value

    def test_set_valid_string(self, chatbot):
        """Test setting a valid string updates the value correctly."""
        valid_string = "openai"
        chatbot.set_model_provider(valid_string)
        assert chatbot.get_model_provider() == valid_string

    def test_set_invalid_string(self, chatbot):
        """Test setting invalid string raises ValueError and does not change the
        value.
        """
        prev_value = chatbot.get_model_provider()
        unsupported_provider = "unsupported_provider"
        with pytest.raises(ValueError):
            chatbot.set_model_provider(unsupported_provider)
        assert chatbot.get_model_provider() == prev_value
        empty_string = ""
        with pytest.raises(ValueError):
            chatbot.set_model_provider(empty_string)
        assert chatbot.get_model_provider() == prev_value

    def test_set_none(self, chatbot, chatbot_with_model):
        """Test setting None updates the value to None."""
        chatbot.set_model_provider(None)
        assert chatbot.get_model_provider() is None
        chatbot_with_model.set_model_provider(None)
        assert chatbot_with_model.get_model_provider() is None
        assert chatbot_with_model.get_model_name() is None


class TestModelNameAttributeManagement:
    """This test class validates model name selection within provider
    constraints, ensuring proper validation of provider-model compatibility and
    dependency management.
    
    Test Coverage:
        - **Provider Dependency**: Model selection requires valid provider
        - **Model Validation**: Only provider-compatible models accepted
        - **Type Validation**: Ensures proper input type checking
        - **State Management**: Confirms attribute updates and consistency
        - **Error Scenarios**: Validates exception handling for invalid inputs

    Critical Dependencies Tested:
        - Model cannot be set without provider
        - Invalid models rejected with ValueError
        - Cross-provider model validation works correctly
    """

    @pytest.fixture
    def chatbot(self):
        """Create a Chatbot instance with a predefined model name."""
        cb = Chatbot()
        cb.set_model_provider("google_genai")
        cb.set_model_name("gemini-2.5-flash")
        return cb
    
    @pytest.fixture
    def chatbot_no_provider(self):
        """Create a Chatbot instance with no predefined model provider."""
        cb = Chatbot()
        return cb

    def test_set_invalid_type(self, chatbot):
        """Test setting an invalid type raises TypeError and does not change 
        the value.
        """
        prev_value = chatbot.get_model_name()
        non_valid_type = 12345
        with pytest.raises(TypeError):
            chatbot.set_model_name(non_valid_type)
        assert chatbot.get_model_name() == prev_value

    def test_set_valid_string(self, chatbot):
        """Test setting a valid string updates the value correctly."""
        valid_string = "gemini-2.5-pro"
        chatbot.set_model_name(valid_string)
        assert chatbot.get_model_name() == valid_string

    def test_set_invalid_string(self, chatbot):
        """Test setting invalid string raises ValueError and does not 
        change the value.
        """
        prev_value = chatbot.get_model_name()
        empty_string = ""
        with pytest.raises(ValueError):
            chatbot.set_model_name(empty_string)
        assert chatbot.get_model_name() == prev_value
        other_provider_model = "gpt-4"
        with pytest.raises(ValueError):
            chatbot.set_model_name(other_provider_model)
        assert chatbot.get_model_name() == prev_value
        unsupported_model = "unsupported_model"
        with pytest.raises(ValueError):
            chatbot.set_model_name(unsupported_model)
        assert chatbot.get_model_name() == prev_value

    def test_set_none(self, chatbot, chatbot_no_provider):
        """Test setting None updates the value to None."""
        chatbot.set_model_name(None)
        assert chatbot.get_model_name() is None
        chatbot_no_provider.set_model_name(None)
        assert chatbot_no_provider.get_model_name() is None

    def test_set_valid_string_with_no_provider(self, chatbot_no_provider):
        """Test setting a valid string with no provider raises ValueError and 
        does not change the value.
        """
        prev_value = chatbot_no_provider.get_model_name()
        valid_string = "gemini-2.5-pro"
        with pytest.raises(ValueError):
            chatbot_no_provider.set_model_name(valid_string)
        assert chatbot_no_provider.get_model_name() == prev_value


class TestModelTemperatureAttributeManagement:
    """Tests model_temperature attribute management in the Chatbot class."""

    @pytest.fixture
    def chatbot(self):
        """Create a Chatbot instance with a predefined model temperature."""
        cb = Chatbot()
        cb.set_model_temperature(0.5)
        return cb

    def test_set_invalid_type(self, chatbot):
        """Test setting an invalid type raises TypeError and does not change 
        the value.
        """
        prev_value = chatbot.get_model_temperature()
        invalid_type = "invalid_type"
        with pytest.raises(TypeError):
            chatbot.set_model_temperature(invalid_type)
        assert chatbot.get_model_temperature() == prev_value

    def test_set_out_of_bounds(self, chatbot):
        """Test setting out-of-bounds values raises ValueError and does not 
        change the value.
        """
        prev_value = chatbot.get_model_temperature()
        delta = 1e-12
        low = 0.0 - delta
        with pytest.raises(ValueError):
            chatbot.set_model_temperature(low)
        assert chatbot.get_model_temperature() == prev_value
        high = 1.0 + delta
        with pytest.raises(ValueError):
            chatbot.set_model_temperature(high)
        assert chatbot.get_model_temperature() == prev_value

    def test_set_valid_values(self, chatbot):
        """Test setting valid values updates the temperature correctly."""
        valid_values = [0, 0.777, 1]
        for value in valid_values:
            chatbot.set_model_temperature(value)
            assert chatbot.get_model_temperature() == float(value)


class TestSystemMessageAttributeManagement:
    """Tests system_message attribute management in the Chatbot class."""

    @pytest.fixture
    def chatbot(self):
        """Create a Chatbot instance with a predefined system message."""
        cb = Chatbot()
        cb.set_system_message("Initial system message.")
        return cb

    def test_set_invalid_type(self, chatbot):
        """Test setting an invalid type raises TypeError and does not change 
        the value.
        """
        prev_value = chatbot.get_system_message()
        invalid_type = 12345
        with pytest.raises(TypeError):
            chatbot.set_system_message(invalid_type)
        assert chatbot.get_system_message() == prev_value

    def test_set_valid_string(self, chatbot):
        """Test setting a valid string updates the value correctly."""
        valid_string = "Updated system message."
        chatbot.set_system_message(valid_string)
        assert chatbot.get_system_message() == valid_string

    def test_set_invalid_string(self, chatbot):
        """Test setting an invalid string raises ValueError and does not change 
        the value.
        """
        prev_value = chatbot.get_system_message()
        empty_string = ""
        with pytest.raises(ValueError):
            chatbot.set_system_message(empty_string)
        assert chatbot.get_system_message() == prev_value

    def test_set_none(self, chatbot):
        """Test setting None raises TypeError and does not change the value."""
        prev_value = chatbot.get_system_message()
        with pytest.raises(TypeError):
            chatbot.set_system_message(None)
        assert chatbot.get_system_message() == prev_value


class TestAPIKeyAttributeManagement:
    """Tests api_key attribute management in the Chatbot class."""

    @pytest.fixture
    def chatbot(self):
        """Create a Chatbot instance with no predefined API key."""
        cb = Chatbot()
        return cb

    @pytest.fixture
    def chatbot_with_api_key(self):
        """Create a Chatbot instance with a predefined API key."""
        cb = Chatbot()
        cb.set_api_key("google_genai", "google_genai_api_key")
        return cb

    def test_set_invalid_type(self, chatbot, chatbot_with_api_key):
        """Test setting an invalid type raises TypeError and does not change 
        the value.
        """
        valid_provider = "google_genai"
        valid_api_key = "valid_api_key"
        invalid_type = 12345
        prev_value = chatbot_with_api_key.get_api_key(valid_provider)
        with pytest.raises(TypeError):
            chatbot_with_api_key.set_api_key(valid_provider, invalid_type)
        assert chatbot_with_api_key.get_api_key(valid_provider) == prev_value
        with pytest.raises(TypeError):
            chatbot.set_api_key(invalid_type, valid_api_key)
        with pytest.raises(TypeError):
            chatbot.set_api_key(invalid_type, invalid_type)

    def test_set_valid_string(self, chatbot, chatbot_with_api_key):
        """Test setting a valid string updates the value correctly."""
        valid_provider = "openai"
        valid_api_key = "valid_api_key"
        chatbot.set_api_key(valid_provider, valid_api_key)
        assert chatbot.get_api_key(valid_provider) == valid_api_key
        new_api_key = "new_google_genai_api_key"
        chatbot_with_api_key.set_api_key("google_genai", new_api_key)
        assert chatbot_with_api_key.get_api_key("google_genai") == new_api_key

    def test_set_invalid_string(self, chatbot, chatbot_with_api_key):
        """Test setting an invalid string raises ValueError and does not change 
        the value.
        """
        invalid_provider = "unsupported_provider"
        empty_provider = ""
        empty_key = ""
        valid_provider = "openai"
        valid_key = "valid_api_key"
        prev_value = chatbot_with_api_key.get_api_key("google_genai")
        with pytest.raises(ValueError):
            chatbot_with_api_key.set_api_key("google_genai", empty_key)
        assert chatbot_with_api_key.get_api_key("google_genai") == prev_value
        with pytest.raises(ValueError):
            chatbot.set_api_key(invalid_provider, valid_key)
        with pytest.raises(ValueError):
            chatbot.set_api_key(empty_provider, valid_key)
        with pytest.raises(ValueError):
            chatbot.set_api_key(valid_provider, empty_key)

    def test_get_unsupported_provider(self, chatbot):
        """Test getting an unsupported provider returns None."""
        assert chatbot.get_api_key("unsupported_provider") is None

    def test_get_no_key_set(self, chatbot):
        """Test getting a key for a provider with no key set returns None."""
        assert chatbot.get_api_key("google_genai") is None

    def test_get_none(self, chatbot):
        """Test getting None as provider returns None."""
        assert chatbot.get_api_key(None) is None

    def test_set_none(self, chatbot, chatbot_with_api_key):
        """Test setting None as provider or API key raises TypeError and does 
        not change the value.
        """
        valid_provider = "openai"
        valid_api_key = "valid_api_key"
        prev_value = chatbot_with_api_key.get_api_key("google_genai")
        with pytest.raises(TypeError):
            chatbot_with_api_key.set_api_key("google_genai", None)
        assert chatbot_with_api_key.get_api_key("google_genai") == prev_value
        with pytest.raises(TypeError):
            chatbot.set_api_key(valid_provider, None)
        with pytest.raises(TypeError):
            chatbot.set_api_key(None, None)
        with pytest.raises(TypeError):
            chatbot.set_api_key(None, valid_api_key)
        assert chatbot.get_api_key(None) is None


class TestGetResponseMethod:
    """This test class validates the get_response method that handles user
    interactions and generates AI responses. It ensures proper message
    handling, validation, and error management.

    Test Coverage:
        - **Message Type Validation**: Ensures only HumanMessage accepted
        - **Error Handling**: Tests exception scenarios and error messages
        - **State Consistency**: Verifies configuration remains intact
    
    Critical Requirements Tested:
        - A valid API key must be provided
        - Input must be valid HumanMessage instance
        - Errors are handled without state corruption
    """

    @pytest.fixture
    def chatbot(self):
        """Create a Chatbot instance with a predefined model provider."""
        cb = Chatbot()
        cb.set_model_provider("google_genai")
        cb.set_model_name("gemini-2.5-flash")
        cb.set_api_key("google_genai", "google_genai_api_key")
        return cb

    def test_get_response_invalid_type(self, chatbot):
        """Test passing an invalid type to get_response raises TypeError and 
        does not change the chat history.
        """
        invalid_type = 12345
        with pytest.raises(TypeError):
            chatbot.get_response(invalid_type)
        assert len(chatbot.get_chat_history()) == 0

    def test_get_response_unexpected_exception(self, chatbot):
        """Test that an unexpected exception is handled and does not change the 
        chat history.
        """
        message = HumanMessage("Valid user input")
        # In this case, we expect an exception to be raised because the api_key
        # is a placeholder and not valid for actual API calls.
        with pytest.raises(Exception):
            chatbot.get_response(message)
        assert len(chatbot.get_chat_history()) == 0
