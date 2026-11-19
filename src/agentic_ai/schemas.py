from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=20, max_length=20_000)
    max_words: int = Field(default=30, ge=5, le=200)


class SummarizeResponse(BaseModel):
    summary: str
    word_count: int
    model: str
