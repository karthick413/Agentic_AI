import json

import pytest
from pydantic import ValidationError

from agentic_ai.models import Ticket

VALID = dict(
    id=1,
    title="Login fails",
    description="Cannot log in since today",
    customer_email="A@Example.com",
    tags=[" Bug ", "bug", "Auth"],
)


def test_valid_ticket_normalizes_tags():
    t = Ticket(**VALID)
    assert t.tags == ["auth", "bug"]


def test_short_title_rejected():
    with pytest.raises(ValidationError):
        Ticket(**{**VALID, "title": "Hi"})


def test_bad_email_rejected():
    with pytest.raises(ValidationError):
        Ticket(**{**VALID, "customer_email": "not-an-email"})


def test_urgent_needs_long_description():
    with pytest.raises(ValidationError):
        Ticket(**{**VALID, "priority": "urgent", "description": "Too short ok"})


def test_json_schema_has_fields():
    assert "title" in Ticket.model_json_schema()["properties"]


def test_invalid_id_rejected():
    with pytest.raises(ValidationError):
        Ticket(**{**VALID, "id": 0})


def test_unknown_priority_rejected():
    with pytest.raises(ValidationError):
        Ticket(**{**VALID, "priority": "critical"})


def test_json_schema_has_fields():
    schema = Ticket.model_json_schema()

    print(json.dumps(schema, indent=4))

    assert "title" in schema["properties"]
