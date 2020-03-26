
from flask import Flask, render_template, request, jsonify
from flask import Flask
import translation_util.interactive_translate as interactive_translation
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

@app.route('/interactive-translation', methods=['POST'])
def translate():
    try:
        inputs = request.get_json(force=True)

        out = interactive_translation.interactive_translation_1(inputs)

        print(out)
        return jsonify(out)
    except Exception as e:
        print(e)
        return jsonify("error")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
    