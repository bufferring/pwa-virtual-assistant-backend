from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[Message]
    temperature: float = 0.4
    max_tokens: int = 1024
    stream: bool = True


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: UsageInfo
