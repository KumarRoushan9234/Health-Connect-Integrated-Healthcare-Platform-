# FROM llama2

# PARAMETER temprature 0.7

# SYSTEM """
# You are an assitent to a doctor based on user input : {user_input} Suggest the Doctor in less than 120 words what disease the patient may suffer from.
# """

# Import from the correct module
from langchain_community.llms import Ollama

# Initialize the Ollama object correctly
ollama = Ollama(base_url="http://localhost:11434", model='llama2')

# Generate a response from the model with the prompt inside a list
response = ollama.generate(["generate a hello world."])
print(response)

