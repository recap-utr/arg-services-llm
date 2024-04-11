import grpc
from arg_services.mining.v1beta import (
    adu_pb2,
    adu_pb2_grpc,
    entailment_pb2,
    entailment_pb2_grpc,
    graph_construction_pb2,
    graph_construction_pb2_grpc,
    major_claim_pb2,
    major_claim_pb2_grpc,
)

tests = range(4)

text = """MOSCOW, March 28. /TASS/. Russia's largest bank, Sberbank, has signed an agreement on selling 100% shares in its Ukrainian affiliation to a consortium that comprises a Latvian bank and a Belarusian company, the bank said in a press release on Monday. "A consortium of investors is purchasing 100% stake PAO Sberbank Ukraine, which is a filial company of PAO Sberbank. The consortium will include Norvik Banka of Latvia and a Belarusian private company," it said. "An appropriate legally binding agreement was signed on Monday." The closure of the transaction is expected before July 2017 after its endorsement by financial and antitrust regulators. The press release said the Ukrainian affiliation had enough reserves to meet the obligations to private and corporate customers likewise. "We hope a decision to sell our filial bank will facilitate the unblocking of its offices and resumption of regular operations and will make it possible for the customers to use without hindrances the services of one of Ukraine's most stable and efficient banks and will lay down the groundwork for its further development," the press release said. British national Said Gutseriyev has become the majority shareholder of the Latvian-Belarusian consortium, which has purchased the Ukrainian affiliation of Sberbank, through a Belarusian company he owns, Latvia's Norvik Banka, the other party to the consortium said in a report. "Said Gutseriyev, a national of the UK, and the Belarusian company he owns has become the majority shareholder of the new consortium," the bank said. "The transaction will enable the Ukrainian customers to enjoy the services based on the European principles of quality, transparency and accessibility and to maintain the levels of technology created by the Sberank of Russia." Along with the main transaction, Norvik Banka, Latvia's bank number seven in terms of assets, has announced a range of steps to cut down its presence in the Russian banking sector - a measure it hopes will help raise the efficiency of investment and eliminate a number political risks arising from the geography of its operations. The report quoted Said Gutseriyev as saying Sberbank Ukraine had a perfect structure, as its previous owner, Sberbank of Russia, had invested hundreds of millions of dollars in the platform of the affiliation. Gutseriyev also said his own experience prompted him that by taking the decision to participate in the consortium the constituent parties made farsighted and promising investment, while the bank as such would be able to make a great leap forward and to implement many advanced projects in Ukraine and in neighboring European countries. Under the pressure of nationalists blocking the Ukrainian affiliations of several banks belonging to Russian lending institutions, President Pyotr Poroshenko imposed sanctions on five banks with Russian funding - Sberbank, Prominvestbank, VTB, BM Bank, and VS Bank - for a period of twelve months as of March 16. The sanctions ban the withdrawal of assets outside of Ukraine, the payment of dividends, as well as the return of interbank deposits and loans and monies from the correspondent accounts of subordinated debts. The restrictions, however, did not embrace transactions between Ukrainian residents and their agents who have accounts in the parent companies."""


def run():
    # address = '136.199.130.136:6790'
    address = "localhost:8888"

    with grpc.insecure_channel(address) as channel:
        if 0 in tests:
            print("Running segmentation test")
            stub = adu_pb2_grpc.AduServiceStub(channel)
            response = stub.Segmentation(adu_pb2.SegmentationRequest(text=text))
            print(response)
            segments = {
                str(idx): segment for idx, segment in enumerate(response.segments)
            }
            # test adu extraction
            print("ADU extraction test")
            segs = {k: v.text for k, v in segments.items()}
            resp2 = stub.Classification(adu_pb2.ClassificationRequest(segments=segs))
            print(resp2)
        if 1 in tests:
            print("Running entailment test")
            stub = entailment_pb2_grpc.EntailmentServiceStub(channel)
            response = stub.Entailments(
                entailment_pb2.EntailmentsRequest(language="en", adus=segments)
            )
            print(response)
            entailments = response.entailments
        if 2 in tests:
            print("Running major claim test")
            stub = major_claim_pb2_grpc.MajorClaimServiceStub(channel)
            response = stub.MajorClaim(
                major_claim_pb2.MajorClaimRequest(
                    language="en", segments=segments, limit=3
                )
            )
            print(response)
            major_claim_ranking = response.ranking
            major_claim_ranking = sorted(
                major_claim_ranking, key=lambda x: x.probability, reverse=True
            )
            major_claim_id = major_claim_ranking[0].id
            print("major_claim_id: ", major_claim_id)
        if 3 in tests:
            print("Running graph construction test")
            stub = graph_construction_pb2_grpc.GraphConstructionServiceStub(channel)
            response = stub.GraphConstruction(
                graph_construction_pb2.GraphConstructionRequest(
                    language="en",
                    adus=segments,
                    major_claim_id=major_claim_id,
                    entailments=entailments,
                )
            )
            print(response)


if __name__ == "__main__":
    run()
