from typing import Mapping

from arg_services.mining.v1beta import adu_pb2, major_claim_pb2, major_claim_pb2_grpc

from .model import MajorClaimGenerator


class MajorClaimServicer(major_claim_pb2_grpc.MajorClaimServiceServicer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.majorclaim_generator = MajorClaimGenerator(*args, **kwargs)

    def _get_ranking(
        self, segments: Mapping[str, adu_pb2.Segment]
    ) -> list[major_claim_pb2.MajorClaimResult]:
        adus = {key: segment.text for key, segment in segments.items()}
        # Achtung: hier kommen Segments an, keine ADUs, welche die OAI Implementierung erwartet
        # text = " ".join(texts)
        majorclaim_probs = self.majorclaim_generator.get_majorclaim_probs(adus)
        results = [
            major_claim_pb2.MajorClaimResult(id=id, probability=prob)
            for id, prob in majorclaim_probs.items()
        ]
        return results

    def MajorClaim(self, request, context):
        segments = (
            request.segments
        )  # maybe should be named more consistently, 02 uses adus
        return major_claim_pb2.MajorClaimResponse(ranking=self._get_ranking(segments))
