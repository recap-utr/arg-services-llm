import arguebuf
from arg_services.mining.v1beta import (
    entailment_pb2,
    graph_construction_pb2,
    graph_construction_pb2_grpc,
)

from .model import GraphConstructor


class GraphConstructionServicer(
    graph_construction_pb2_grpc.GraphConstructionServiceServicer
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.graph_constructor = GraphConstructor(*args, **kwargs)

    _type_mapping = {
        entailment_pb2.EntailmentType.ENTAILMENT_TYPE_ENTAILMENT: "support",
        entailment_pb2.EntailmentType.ENTAILMENT_TYPE_CONTRADICTION: "attack",
    }

    def _construct_graph(self, adus, entailments, major_claim_id):
        adu_texts = {id: adu.text for id, adu in adus.items()}
        entailment_dicts = [
            {
                "source": entailment.premise_id,
                "target": entailment.claim_id,
                "type": self._type_mapping[entailment.type],
            }
            for entailment in entailments
        ]
        generated_graph = self.graph_constructor.build_graph(
            adu_texts, entailment_dicts, major_claim_id
        )
        return generated_graph

    def GraphConstruction(self, request, context):
        adus = request.adus
        entailments = request.entailments
        major_claim_id = request.major_claim_id
        generated_graph = self._construct_graph(adus, entailments, major_claim_id)
        if generated_graph is None:
            return graph_construction_pb2.GraphConstructionResponse(
                graph=arguebuf.dump.protobuf(arguebuf.Graph())
            )
        return graph_construction_pb2.GraphConstructionResponse(
            graph=arguebuf.dump.protobuf(generated_graph)
        )
