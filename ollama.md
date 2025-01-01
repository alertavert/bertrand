# Ollama Fundamentals
[Course on YouTube](https://youtu.be/GWB9ApTPTv4) and 
[code](https://github.com/pdichone/ollama-fundamentals)

### Useful links
- [Ollama](https://ollama.com/)
- [Msty App](https://msty.app)
- [Streamlit](https://streamlit.io)
- [AI Recruitment Agency Demo code](https://github.com/pdichone/swarm-writer-agents/tree/main/ai-recruiter-agency)

## Running a model
To run a model, just use:
```shell
ollama run nemotron
```

## Llava
A multi-modal trained model that combines a vision encoder; in 7B, 13B and 34B sizes:

```shell
ollama run llava:34b
```

## Modelfile
It is used to create a varian of a model, with parameters specified:

```model
FROM llama3.2

PARAMETER temperature 0.3

SYSTEM """
    You are a coding assistant and will respond to queries with
    code snippets. You will be able to generate code snippets
    in Go and the necessary shell commands to run them.
    Alongside the code you will provide simple, short explanations
    of the code snippets.
"""
```

The model is actually created with:
```shell
└─( ollama create Majordomo -f ./Modelfile
```
and then run as usual:
```shell
└─( ollama run Majordomo
```

## REST API
The backend runs at `localhost:11434` so it can be queried with a POST request to `localhost:11434/ollama` with the following JSON body:
```shell
└─( http POST :11434/api/generate \
    model="Majordomo" \
    prompt="What is the best way to sort a list in Python" \
    stream=false
```
It also offers the `/api/chat` endpoint to chat with the model; and we can specify the output format
as JSON with `format=json`.

See the [docs](https://github.com/ollama/docs/api.md) for more information.

## Msty Application

[Msty](https://msty.app) is a web application that uses the Ollama API to generate code snippets and explanations for a given prompt. It is a simple, easy-to-use interface 
that allows users to interact with the model without having to use the command line.

## Ollama Python Package

```shell
pip install ollama
```
then use it as shown in [`use_model.py`](examples/use_model.py_model.py).

['pdf-rag-streamlit'](examples/pdf-rag-streamlit.pyeamlit.py) is a Streamlit app that uses the Ollama API to 
analyzed a PDF and answers questions about it.
