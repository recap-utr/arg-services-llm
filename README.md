# ArgMining LLM

- Copy .env.sample to .env and input your OpenAI API Key.
- Use `poetry install` in the base directory to install the project's dependencies.
- Start the gRPC Server with `poetry run python -m argmining.server`.
- Start the gRPC Client with `poetry run python -m argmining.client` to run a test case.

This project contains one folder for each of the argument extraction pipeline steps: entailment (argument relation classification), adu extraction, major claim prediction and graph construction.
In each of these folders, a gRPC Servicer is defined, which handles the gRPC types, i.e., mapping a gRPC request to a gRPC response. Internally, it calls the respective model.py, which uses the OpenAI API to implement the actual processing.
