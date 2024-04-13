import os
import json
from flask import Flask, jsonify
from flask_cors import CORS
from markupsafe import escape

app = Flask(__name__)
cors = CORS(app, origins="*")
results_path = './../results'

@app.route("/api/experiments", methods=['GET'])
def get_experiments():
    contents = os.listdir(results_path)
    directories = [item for item in contents if os.path.isdir(os.path.join(results_path, item))]
    return jsonify(directories)

@app.route("/api/experiments/<exp_id>")
def get_experiment_data(exp_id):
    exp_path = f'{results_path}/{escape(exp_id)}'
    folder_content = os.listdir(exp_path);

    exp_data = []

    for f in folder_content:
        if f.endswith('.json') and f.split('.')[0].isdigit():
            with open(f'{exp_path}/{f}', 'r') as json_file:
                data = json.load(json_file)
                exp_data.append({
                    "round": data['rnd'],
                    "global_loss": data['global_loss'],
                    "global_accuracy": data['global_accuracy']
                })

    return jsonify(exp_data)

if __name__ == "__main__":
    app.run(debug=True, port=44555)
