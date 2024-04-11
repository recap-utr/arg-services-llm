from typing import Mapping

from arg_services.mining.v1beta import adu_pb2, adu_pb2_grpc
from nltk.tokenize import word_tokenize

from .model import Extractor


class ExtractionServicer(adu_pb2_grpc.AduServiceServicer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.extractor = Extractor(*args, **kwargs)

    def _divide_segments(self, text: str) -> list[adu_pb2.Segment]:
        return [
            adu_pb2.Segment(text=segment.text, start=segment.start, end=segment.end)
            for segment in self.extractor.divide_segments(text)
        ]

    def _classify_segments(self, segments: Mapping[str, str]) -> list[adu_pb2.Adu]:
        adus = []
        for key, text in segments.items():
            evaluation_response = self.extractor.evaluation(text)
            for idx, adu in enumerate(evaluation_response):
                tokens = [
                    adu_pb2.Token(text=token) for token in word_tokenize(adu.text)
                ]
                adus.append(adu_pb2.Adu(segment_id=f"{key}-{idx}", tokens=tokens))
        return adus

    def Segmentation(self, request, context):
        segments = self._divide_segments(request.text)
        return adu_pb2.SegmentationResponse(segments=segments)

    def Classification(self, request, context):
        adus = self._classify_segments(request.segments)
        return adu_pb2.ClassificationResponse(adus=adus)
