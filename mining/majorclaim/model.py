import json
import os
from typing import Mapping

import openai


class MajorClaimGenerator:
    _system_prompt: str = """
        The user will provide a list of argumentative discourse units (ADUs).
        Your task is to predict the probability if a certain ADU is the major claim / conclusion of an argument for every ADU.
        The major claim will subsequently be used as the root node of an argument graph.
    """

    _identified_major_claim_schema: dict = {
        "type": "object",
        "required": ["id", "probability"],
        "properties": {
            "id": {
                "type": "string",
                "description": "The ID of the ADU",
            },
            "probability": {
                "type": "number",
                "description": "The probability that the ADU is the major claim",
            },
        },
    }

    def __init__(self, model="gpt-4-turbo-preview"):
        self.model = model
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_majorclaim_probs(self, adus: Mapping[str, str]) -> Mapping[str, float]:
        completion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": json.dumps({"adus": adus})},
            ],
            functions=[
                {
                    "name": "major_claim_rating",
                    "description": "Identify the major claim / conclusion of an argument",
                    "parameters": {
                        "title": "Major Claim Rating",
                        "description": "Identify the major claim / conclusion of an argument",
                        "type": "object",
                        "required": ["major_claim_probabilities"],
                        "properties": {
                            "major_claim_probabilities": {
                                "type": "array",
                                "items": self._identified_major_claim_schema,
                            },
                        },
                    },
                }
            ],
            function_call={"name": "major_claim_rating"},
        )
        message = completion.choices[0].message.function_call
        if message is None:
            return {}
        arguments = message.arguments

        major_claim_probs = json.loads(arguments).get("major_claim_probabilities", [])
        return {mc["id"]: mc["probability"] for mc in major_claim_probs}
