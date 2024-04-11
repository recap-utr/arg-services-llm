from typing import Mapping

from arg_services.mining.v1beta import adu_pb2, entailment_pb2, entailment_pb2_grpc

from .model import EntailmentClassifier


class EntailmentServicer(entailment_pb2_grpc.EntailmentServiceServicer):
    _entailment_types = {
        "support": entailment_pb2.EntailmentType.ENTAILMENT_TYPE_ENTAILMENT,
        "attack": entailment_pb2.EntailmentType.ENTAILMENT_TYPE_CONTRADICTION,
        "neut": entailment_pb2.EntailmentType.ENTAILMENT_TYPE_NEUTRAL,
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.entailment_classifier = EntailmentClassifier(*args, **kwargs)

    def _get_entailments(
        self, adus: Mapping[str, adu_pb2.Segment]
    ) -> list[entailment_pb2.Entailment]:
        adu_texts = {id: adu.text for id, adu in adus.items()}
        entailments = self.entailment_classifier.predict(adu_texts)
        entailments = [
            entailment_pb2.Entailment(
                premise_id=entailment.source,
                claim_id=entailment.target,
                type=self._entailment_types[entailment.type],
            )
            for entailment in entailments
        ]
        return entailments

    def Entailments(self, request: entailment_pb2.EntailmentsRequest, context):
        return entailment_pb2.EntailmentsResponse(
            entailments=self._get_entailments(request.adus)
        )
