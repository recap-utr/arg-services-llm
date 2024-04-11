import json
import os

import openai
import spacy
from pydantic import BaseModel

# model source code is adjusted from "Fine-Grained Argument Unit Recognition and Classification" by Trautmann et al. (DOI: https://doi.org/10.1609/aaai.v34i05.6438)


class Segment(BaseModel):
    text: str
    start: int
    end: int


class ADU(BaseModel):
    text: str


class Extractor:
    _system_prompt: str = """
    The user will provide a long text that contains a set of arguments.
    Your task is to identify all argumentative discourse units (ADUs) in the text.
    They will subsequently be used to construct a graph.
    The user will have the chance to correct the graph, so DO NOT change any text during this step.
    You shall only EXTRACT the ADUs from the text.
    """

    _extacted_adu_schema: dict = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {
                "type": "string",
                "description": "The text of the ADU",
            },
        },
    }

    def __init__(self, model="gpt-4-turbo-preview"):
        self.model = model
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def divide_segments(self, text: str) -> list[Segment]:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return [
            Segment(text=sent.text, start=sent.start, end=sent.end)
            for sent in doc.sents
        ]

    def evaluation(self, text) -> list[ADU]:
        completion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": text},
            ],
            functions=[
                {
                    "name": "extract_adus",
                    "description": "Extract a set of argumentative discourse units (ADUs) from a resource",
                    "parameters": {
                        "title": "ADU extraction",
                        "description": "Extract a set of argumentative discourse units (ADUs) from a resource",
                        "type": "object",
                        "required": ["adus"],
                        "properties": {
                            "adus": {
                                "type": "array",
                                "items": self._extacted_adu_schema,
                            },
                        },
                    },
                }
            ],
            function_call={"name": "extract_adus"},
        )
        message = completion.choices[0].message.function_call
        if message is None:
            return []
        arguments = message.arguments
        adus = json.loads(arguments).get("adus", [])
        # convert adus list of dicts to list of ADU objects
        adus = [ADU(**adu) for adu in adus]
        return adus
