# S.A.B.E.R.

[![codecov](https://codecov.io/github/rafaeljc/saber/graph/badge.svg?token=NN7LV689Q2)](https://codecov.io/github/rafaeljc/saber)

## Description
S.A.B.E.R. (Smart Assistant for Better Engineering and Research) is an AI-powered assistant designed to help engineers and researchers solve problems and find information more efficiently. It uses Streamlit and LangChain to provide a conversational interface using large language models (LLMs), allowing users to interact naturally and access advanced AI features through a user-friendly web application.

## Why?
S.A.B.E.R. was created to streamline the workflow of engineers and researchers who often need to search for information and experiment with different LLMs. With S.A.B.E.R., you can:
- Select and use multiple LLMs in a single tool.
- Adjust key LLM settings (context, temperature, number of tokens, etc.) through an intuitive interface.
- Use Retrieval-Augmented Generation (RAG) to improve the quality and relevance of model responses.

By integrating these features, S.A.B.E.R. empowers users to focus on their main tasks, rather than on tool management or manual research.

## Quick Start
### Download
You can either download the latest release from the [Releases page](https://github.com/rafaeljc/saber/releases) or clone the repository using Git:
```bash
git clone https://github.com/rafaeljc/saber.git
cd saber
```
### Install the application using pip
```bash
pip install .
```
### Run the application
```bash
saber
```
### Access the application in your browser
Visit: [http://localhost:8501](http://localhost:8501)

## Contributing
To get started:

1. Install uv:
   ```bash
   pip install uv
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/rafaeljc/saber.git
   cd saber
   ```
3. Install dependencies:
   ```bash
   uv sync
   ```
4. Lint your code with Ruff:
   ```bash
   uv run ruff check --fix .
   uv run ruff format .
   ```
5. Run the tests:
   ```bash
   uv run pytest -n auto
   ```
6. Install and run the application:
   ```bash
   uv run pip install -e .
   uv run saber
   ```

If you have any questions or suggestions, please open an issue or pull request.
