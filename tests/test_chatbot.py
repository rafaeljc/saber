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
