def format_text(text):
    # Split the text by semicolons
    parts = text.split(';')

    # Add a newline and bullet for each part
    formatted_text = ""
    for part in parts:
        formatted_text += f"• {part.strip()}\n"

    return formatted_text.strip()

# Input text
text = """
1. Flu - Fever, chills, muscle aches, fatigue; antiviral medications (amantadine and rimantadine) and pain relievers like acetaminophen or ibuprofen; hydration with water and fluids; rest; over-the-counter cold and flu treatments for kids; 
2. Bronchitis - cough, chest pain, shortness of breath, fatigue; expectorants (such as cough suppressants) to help clear phlegm from the lungs; oxygen therapy if breathing is difficult; antibiotics; 
3. Pneumonia - fever, chills, muscle aches, fatigue; expectorants and cough suppressants; antibiotics; oxygen therapy if breathing is difficult; 
4. Back pain - muscle aches, fatigue, nausea or vomiting; rest; over-the-counter pain relievers like acetaminophen or ibuprofen; heat or cold packs; physical therapy; 
5. Hypothermia - shivering and cold; warm clothing; blankets; rehydration solutions; medications to help raise body temperature (like oral rehydration solution)
"""

# Format the text
formatted_text = format_text(text)

# Print the formatted text
print(formatted_text)
