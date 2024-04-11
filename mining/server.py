from concurrent import futures

import grpc
from arg_services.mining.v1beta import (
    adu_pb2_grpc,
    entailment_pb2_grpc,
    graph_construction_pb2_grpc,
    major_claim_pb2_grpc,
)

from mining.entailment.create_servicer import EntailmentServicer
from mining.extraction.create_servicer import ExtractionServicer
from mining.graphconstruction.create_servicer import GraphConstructionServicer
from mining.majorclaim.create_servicer import MajorClaimServicer


def serve():
    global config, entailment_classifier, majorclaim_generator, nlp, graph_constructor
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    adu_pb2_grpc.add_AduServiceServicer_to_server(ExtractionServicer(), server)
    entailment_pb2_grpc.add_EntailmentServiceServicer_to_server(
        EntailmentServicer(),
        server,
    )
    major_claim_pb2_grpc.add_MajorClaimServiceServicer_to_server(
        MajorClaimServicer(),
        server,
    )
    graph_construction_pb2_grpc.add_GraphConstructionServiceServicer_to_server(
        GraphConstructionServicer(),
        server,
    )
    server.add_insecure_port("[::]:50500")
    server.start()
    print("Server started")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
