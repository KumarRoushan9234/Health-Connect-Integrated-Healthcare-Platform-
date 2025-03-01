from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS
import pickle


file_path = 'static/langchain_model.pkl'
with open(file_path, 'rb') as f:
    loaded_model = pickle.load(f)

print("Model loaded successfully.")

def fix_indentation(input_string):
    lines = input_string.split('\n')

    indentation_level = 0
    converted_code = ''
    stack = []  # Stack to keep track of indentation levels
    for line in lines:
        line = line.strip() 
        if line:
            if line.endswith(':'):
                converted_code += '    ' * indentation_level + line + '\n'
                stack.append(indentation_level)  # Push current level to stack
                indentation_level += 1
            elif line.startswith('return'):
                indentation_level = stack.pop()  # Pop previous level from stack
                converted_code += '    ' * indentation_level + line + '\n'
            else:
                converted_code += '    ' * indentation_level + line + '\n'
        else:
            converted_code += '\n'

    return converted_code

def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():  # Change function name to 'index'
    return render_template('convert.html')

# @app.route('/success/<string:output>',endpoint="success")
# def success(output):
#     return render_template("result.html", output=output)

@app.route('/convert', methods=['POST'])
def convert_code():
    if request.method == "POST":
        query = request.json.get('query')

        print("-" * 40)
        print(query)

        docs = loaded_model.similarity_search(query)

        print("-" * 40)
        print(type(docs))

        para = wrap_text_preserve_newlines(str(docs[0].page_content))
        newpara = para.replace("\n", " ")
        lines = newpara.split("NEW_LINE")
        converted_code = " "
        indentation_stack = []

        for line in lines:
            if line.startswith(" INDENT"):
                indentation_stack.append("    ")
                line = line.replace(" INDENT", "")
            elif line.startswith(" DEDENT"):
                dedent_count = line.count(" DEDENT")
                while dedent_count > 0 and indentation_stack:
                    indentation_stack.pop()
                    dedent_count -= 1
                line = line.replace(" DEDENT", "")
            line = "".join(indentation_stack) + line
            converted_code += line + "\n"

        
        converted_code = fix_indentation(converted_code)
        print("-" * 40)
        print(converted_code)
        # print(type(converted_code))

        # return render_template("result.html", output=converted_code)
        return jsonify({'converted_code': converted_code})



if __name__ == '__main__':
    app.run(debug=True, port=8001) 






