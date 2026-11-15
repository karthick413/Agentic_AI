from datetime import UTC, datetime
from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"


class Ticket(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    id: int = Field(gt=0)
    title: str = Field(min_length=5, max_length=120)
    description: str = Field(min_length=10)
    priority: Priority = Priority.medium
    customer_email: EmailStr
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, v: list[str]) -> list[str]:
        return sorted({t.strip().lower() for t in v if t.strip()})

    @model_validator(mode="after")
    def urgent_needs_detail(self):
        if self.priority == Priority.urgent and len(self.description) < 30:
            raise ValueError("urgent tickets need a description of 30+ chars")
        return self
