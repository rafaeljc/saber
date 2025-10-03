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

    def test_set_invalid_type(self, chatbot):
        """Test setting an invalid type raises TypeError and does not change 
        the value.
        """
        prev_value = chatbot.get_model_provider()
        non_valid_type = 12345
        with pytest.raises(TypeError):
            chatbot.set_model_provider(non_valid_type)
        assert chatbot.get_model_provider() == prev_value

    def test_set_valid_string_type(self, chatbot):
        """Test setting a valid string type updates the value correctly."""
        valid_string = "openai"
        chatbot.set_model_provider(valid_string)
        assert chatbot.get_model_provider() == valid_string

    def test_set_invalid_string_values(self, chatbot):
        """Test setting invalid string values raises ValueError and does not 
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

    def test_set_none_type(self, chatbot):
        """Test setting None updates the value to None."""
        chatbot.set_model_provider(None)
        assert chatbot.get_model_provider() is None
