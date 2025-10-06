import pytest

from saber.chatbot import Chatbot


class TestModelProviderAttributeManagement:
    """Tests model_provider attribute management in the Chatbot class."""

    @pytest.fixture
    def chatbot(self):
        """Create a Chatbot instance with a predefined model provider."""
        cb = Chatbot()
        cb.set_model_provider("google_genai")
        return cb
    
    @pytest.fixture
    def chatbot_with_model(self):
        """Create a Chatbot instance with a predefined model name."""
        cb = Chatbot()
        cb.set_model_provider("google_genai")
        cb.set_model_name("gemini-2.5-flash")
        return cb

    def test_set_invalid_type(self, chatbot):
        """Test setting an invalid type raises TypeError and does not change 
        the value.
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
        """Test setting invalid string raises ValueError and does not 
        change the value.
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
    """Tests model_name attribute management in the Chatbot class."""

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
