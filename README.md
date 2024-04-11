# LLM-based Argumentation Services

- Make sure to set the environment variable `OPENAI_API_KEY` to your OpenAI API key.
- Use `poetry install` in the base directory to install the project's dependencies.

## Argument Mining

- Start the gRPC Server with `poetry run python -m mining.server`.
- Start the gRPC Client with `poetry run python -m mining.client` to run a test case.

This project contains one folder for each of the argument extraction pipeline steps: entailment (argument relation classification), adu extraction, major claim prediction and graph construction.
In each of these folders, a gRPC Servicer is defined, which handles the gRPC types, i.e., mapping a gRPC request to a gRPC response. Internally, it calls the respective model.py, which uses the OpenAI API to implement the actual processing.

## Quality Assessment

- Start the gRPC Server with `poetry run python -m quality.server`.

The service currently only supports quality explanation.

## Argument Ranking

- Start the gRPC Server with `poetry run python -m ranking.server`.

The service currently only supports fine-granular clustering.
