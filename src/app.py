from flask import Flask, request, render_template, flash
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import io
import base64
import urllib

app = Flask(__name__, template_folder='templates')
app.secret_key = 'secret'
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../models')
models = {f.split('.')[0]: tf.keras.models.load_model(os.path.join(models_dir, f)) for f in os.listdir('../models') if
          f.endswith('.h5')}


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    model_selected = None
    plot_url = None
    if request.method == 'POST':
        try:
            model_selected = request.form['model']
            if model_selected not in models:
                flash("Modèle non trouvé", 'error')
                return render_template('index.html', result=result, model=model_selected, models=models.keys())
        except Exception as e:
            flash("Erreur lors de la sélection du modèle", 'error')
            return render_template('index.html', result=result, model=model_selected, models=models.keys())

        try:
            values = np.array([float(x) for x in request.form['values'].split(',') if x])
            if values.size == 0:
                flash("Aucune valeur fournie", 'error')
                return render_template('index.html', result=result, model=model_selected, models=models.keys())
            values = values.reshape(1, -1, 1)
        except ValueError:
            flash("Entrée invalide : toutes les valeurs doivent être des nombres", 'error')
            return render_template('index.html', result=result, model=model_selected, models=models.keys())

        try:
            predictions = models[model_selected].predict(values)
            result = int(predictions[0][0] >= 0.5)
        except Exception as e:
            flash("Erreur lors de la prédiction", 'error')
            return render_template('index.html', result=result, model=model_selected, models=models.keys())

        # Sauvegarde du graphique
        plt.figure()
        plt.plot(values[0])
        plt.title(f"Patient's data")
        plt.xlabel('Time')
        plt.ylabel('Value')
        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)
        plt.close()
        plot_url = base64.b64encode(bytes_image.getvalue()).decode()

    return render_template('index.html', result=result, model=model_selected, models=models.keys(), plot_url=plot_url)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
