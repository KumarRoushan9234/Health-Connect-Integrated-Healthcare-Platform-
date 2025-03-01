import ollama

# Ensure the model name is correct and available
model_name = 'llama3.1'

# User input
user_input = "i am suffering from fever, itching, cold, headache"

# Run the chat with the specified model
response = ollama.chat(model=model_name, messages=[
  {
    'role': 'user',
    'content': f'''This is the user's input: '{user_input}'. I want you to analyze it and list all these aspects [Mood, Trigger, Focus, Personality, Mental profile, Environment, Habit, Song Recommendation (English, Hindi), Analysis, Personalized Solution] line by line according to the user's feelings. If you cannot get any answer for a category, just return 'Nan'.''',
  },
])

# Print the response
print(response)
