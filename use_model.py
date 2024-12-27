import ollama

# Get info about the model
res = ollama.show(model='Majordomo')
print(res.model_dump_json(indent=2))

# Query the model
res = ollama.generate(
    model='Majordomo',
    prompt="""
        Show me a short snippet of Python code to send a POST request to a URL 
        and parse the body response.
        The response is in JSON format and your code should print out the contents
        of the "data" key, if it exists.
        If the "data" key does not exist, print out an error message.
        Make sure to handle correctly error response codes, such as 404 or 500. 
        """,
)
print(res["response"])
