
from enum import StrEnum
from typing import List, Dict


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"

    def __repr__(self):
        if self == Role.USER:
            return "USER"
        elif self == Role.ASSISTANT:
            return "ASSISTANT"

class Conversation:
    def __init__(self):
        self._messages: List[Dict[Role, str]] = []

    def add_message(self, role: Role, message: str):
        self._messages.append({role: message})

    @property
    def messages(self) -> List[Dict[Role, str]]:
        return self._messages

    def __len__(self):
        return len(self._messages)

    def reset(self):
        """Reset the conversation."""
        self._messages = []

    def truncate(self, n: int):
        """Truncate the conversation to the last `n` messages."""
        self._messages = self._messages[-n:]

    def head(self, n: int = 5):
        """Return the first `n` messages."""
        return self._messages[:n]

    def tail(self, n: int = 5):
        """Return the last `n` messages."""
        return self._messages[-n:]
