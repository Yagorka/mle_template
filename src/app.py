from flask import Flask, request, render_template
from predict import Predictor
import traceback
from logger import Logger
import sys
import pandas as pd

SHOW_LOG = True

logger = Logger(SHOW_LOG)
log = logger.get_logger(__name__)

app = Flask(__name__)
app.debug = True

log.info("Loading SVM model")
try:
    svm_model = Predictor()
except Exception:
    log.error(traceback.format_exc())
    sys.exit(1)

@app.route("/", methods=["POST", "GET"]) 
def render_send():

    page_context = None
    if request.method == 'POST':
        data = [list(map(float, request.form.get("X").split(',')))]
        #print(data)

        page_context = {
                'X': data,
                'X_predict': 'Rock' if svm_model.predict_web(data) == ['R'] else 'Metal'
            }
        log.info(f"Predict class from '{data}' is {page_context['X_predict']}")

    return render_template('index.html', context=page_context)
