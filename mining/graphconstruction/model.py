import json
import os
from typing import Mapping

import openai
from arguebuf import AtomNode, Edge, Graph, SchemeNode
from arguebuf.model.scheme import Attack, Support


class GraphConstructor:
    _system_prompt: str = """
    The user will provide a list of argumentative discourse units (ADUs), the relations between these and the ID of the major claim.
    The user provided relations do not include relations between the major claim and other ADUs. Generating these relations is your task.
    You shall create a valid hierarchical graph with the major claim being the root node (i.e., it should have no outgoing relations, only incoming ones).
    You shall only add relations, not remove or modify existing ones.
    Flat graphs (i.e., all ADUs directly connected to the major claim directly) are discouraged.
    There should be no cycles in the graph and no orphaned ADUs.
    """

    _predicted_relation_schema: dict = {
        "type": "object",
        "required": ["source", "target", "type", "explanation"],
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

    _type_mapping = {
        "support": Support.DEFAULT,
        "attack": Attack.DEFAULT,
    }

    def __init__(self, model="gpt-4-turbo-preview"):
        self.model = model
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def build_graph(
        self,
        adus: Mapping[str, str],
        relations: list[dict[str, str]],
        major_claim_id: str,
    ) -> Graph | None:
        completion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "adus": adus,
                            "relations": relations,
                            "major_claim_id": major_claim_id,
                        }
                    ),
                },
            ],
            functions=[
                {
                    "name": "predict_relations",
                    "description": "Predict relations between the major claim and other argumentative discourse units",
                    "parameters": {
                        "title": "Relation Generation",
                        "description": "Predict relations between the major claim and other argumentative discourse units",
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
            return None
        arguments = message.arguments
        relations = json.loads(arguments).get("relations", [])
        graph = Graph()
        i_nodes = {}
        for adu_id, adu_text in adus.items():
            i_node = AtomNode(text=adu_text)
            i_nodes[adu_id] = i_node
            graph.add_node(i_node)
        for relation in relations:
            source_node = i_nodes.get(relation["source"], None)
            target_node = i_nodes.get(relation["target"], None)
            if source_node is None or target_node is None:
                continue
            relation_type = self._type_mapping[relation["type"]]
            s_node = SchemeNode(relation_type)
            graph.add_node(s_node)
            graph.add_edge(Edge(source_node, s_node))
            graph.add_edge(Edge(s_node, target_node))
        return graph
