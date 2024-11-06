### Intro to Jeff's fork

This is my fork of an awesome workshop repository for building generative AI apps started by [@hugobowne](https://github.com/hugobowne).
I've made the apps more robust to local model availability and made the conversation logs more comprehensive (input file and model used and time taken for the query).

Summary of the changes I've made:

- The conversation app (`4-app-convo.py`) uses OpenAIEmbedding instead of a local embedding model if the OpenAI option is selected.
  The imports of Ollama and HuggingFaceEmbedding are made conditional in case a local model isn't available.
  This situation is indicated in the app by a message, and the radio selector is limited to only OpenAI.
- The logging app (`5-app-convo-log.py`) logs the PDF file name and LLM name.
  The user's query is logged before running the query so that model timing can be calculated from the logs.
- The apps use `llama_index.core.Settings` to specify the models.
  If OpenAI is selected, the LLM and embedding models are `gpt-4o-mini` and `text-embedding-3-small`, respectively.
  The local model (Ollama) is `llama3.2`.

Requirements cheatsheet:

```
# For 1-app-query.py
pip install llama-index
# For 2-app-front-end.py
pip install gradio
pip install PyMuPDF
# For 3-app-local.py
pip install llama-index-llms-ollama
pip install llama-index-embeddings-huggingface
curl -fsSL https://ollama.com/install.sh | sh # Run as root
ollama serve         # Do this before running the app
ollama run llama3.2  # Only needed once to download the model
# To explore the SQLite database created in 5-app-convo-log.py
pip install datasette
```

The original description is below!

___

This repo is a WIP. We'll teach our first iteration of this workshop in Nov 2024 at the [MLOps World and Generative AI World Conference](https://generative-ai-summit.com/).

## Description:
This workshop is designed to equip software engineers with the skills to build and iterate on generative AI-powered applications. Participants will explore key components of the AI software development lifecycle through first principles thinking, including prompt engineering, monitoring, evaluations, and handling non-determinism. The session focuses on using multimodal AI models to build applications, such as querying PDFs, while providing insights into the engineering challenges unique to AI systems. By the end of the workshop, participants will know how to build a PDF-querying app, but all techniques learned will be generalizable for building a variety of generative AI applications.

If you're a data scientist, machine learning practitioner, or AI enthusiast, this workshop can also be valuable for learning about the software engineering aspects of AI applications, such as lifecycle management, iterative development, and monitoring, which are critical for production-level AI systems.

## What You'll Learn:
- How to integrate AI models and APIs into a practical application.
- Techniques to manage non-determinism and optimize outputs through prompt engineering.
- How to monitor, log, and evaluate AI systems to ensure reliability.
- The importance of handling structured outputs and using function calling in AI models.
- The software engineering side of building AI systems, including iterative development, debugging, and performance monitoring.
- Practical experience in building an app to query PDFs using multimodal models.


## Workshop Prerequisite Knowledge:
- Basic programming knowledge in Python.
- Familiarity with REST APIs.
- Experience working with Jupyter Notebooks or similar environments (preferred but not required).
- No prior experience with AI or machine learning is required.
- Most importantly, a sense of curiosity and a desire to learn!

If you have a background in data science, ML, or AI, this workshop will help you understand the software engineering side of building AI applications.
