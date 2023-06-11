from flask import Flask, request, render_template, flash
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import io
import base64
import urllib

app = Flask(__name__, template_folder='templates')
app.secret_key = 'secret!'
models_dir = os.path.join(os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
), 'models')
print(models_dir)
models = {f.split('.')[0]: tf.keras.models.load_model(os.path.join(models_dir, f)) for f in os.listdir(models_dir) if
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
            # recuperer les valeurs
            datas = request.form['values']
            occurrences = {separator: datas.count(separator) for separator in [',', '\t', ' ']}
            most_used_spearator = max(occurrences, key=occurrences.get)
            values = np.array([float(x) for x in datas.split(most_used_spearator) if x])
            print(values)
            if values.size == 0:
                flash("Aucune valeur fournie", 'error')
                return render_template('index.html', result=result, model=model_selected, models=models.keys())
            values = values.reshape(1, -1, 1)
        except ValueError:
            flash("Entrée invalide : toutes les valeurs doivent être des nombres", 'error')
            return render_template('index.html', result=result, model=model_selected, models=models.keys())

        try:
            predictions = models[model_selected].predict(values)
            print(predictions)
            result = f"sain avec une certitude de {int((predictions[0][1]) * 100)}%" if predictions[0][1] >= 0.5 \
                else f"malade avec une certitude de {int((1 - predictions[0][1]) * 100)}%"
        except Exception as e:
            flash("Erreur lors de la prédiction", 'error')
            return render_template('index.html', result=result, model=model_selected, models=models.keys())

        # Sauvegarde du graphique
        plt.figure()
        plt.plot(values[0])
        plt.title(f"Donnée du patient")
        plt.xlabel('Temps')
        plt.ylabel('Valeur')
        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)
        plt.close()
        plot_url = base64.b64encode(bytes_image.getvalue()).decode()

    return render_template('index.html', result=result, model=model_selected, models=models.keys(), plot_url=plot_url)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
