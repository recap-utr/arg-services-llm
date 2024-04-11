import json
import os
from typing import Mapping

import openai
from pydantic import BaseModel


class Relation(BaseModel):
    source: str
    target: str
    type: str


class EntailmentClassifier:
    _system_prompt: str = """
    The user will provide a list of argumentative discourse units (ADUs).
    Your task is to predict sensible relations in the form of support/attack between them.
    You shall produce a valid argument graph with every ADU connected to at least one other ADU (no orphaned ADUs).
    There should be no cycles.
    """

    _predicted_relation_schema: dict = {
        "type": "object",
        "required": ["source", "target", "type"],
        "properties": {
            "source": {
                "type": "string",
                "description": "The ID of the source ADU",
            },
            "target": {
                "type": "string",
                "description": "The ID of the target ADU",
            },
            "type": {
                "type": "string",
                "description": "The type of the relation",
                "enum": ["support", "attack"],
            },
        },
    }

    def __init__(self, model="gpt-4-turbo-preview"):
        self.model = model
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def predict(self, adus: Mapping[str, str]) -> list[Relation]:
        completion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": json.dumps({"adus": adus})},
            ],
            functions=[
                {
                    "name": "predict_relations",
                    "description": "Predict relations between argumentative discourse units",
                    "parameters": {
                        "title": "Relation Generation",
                        "description": "Predict relations between argumentative discourse units",
                        "type": "object",
                        "required": ["relations"],
                        "properties": {
                            "relations": {
                                "type": "array",
                                "items": self._predicted_relation_schema,
                            },
                        },
                    },
                }
            ],
            function_call={"name": "predict_relations"},
        )
        message = completion.choices[0].message.function_call
        if message is None:
            return []
        arguments = message.arguments
        relations = json.loads(arguments).get("relations", [])
        relations = [
            Relation(
                source=relation["source"],
                target=relation["target"],
                type=relation["type"],
            )
            for relation in relations
        ]
        return relations
