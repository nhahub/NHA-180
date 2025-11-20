import joblib
import numpy as np
import json
import gradio as gr

MODEL_PATH = "model_rf_pipeline.joblib"
COLS_PATH = "cols.json"

model = joblib.load(MODEL_PATH)
with open(COLS_PATH, "r") as f:
    FEATURES_ORDER = json.load(f)

def predict_human(**kwargs):
    try:
        values = [float(kwargs.get(col, 0)) for col in FEATURES_ORDER]
    except Exception as e:
        return f"Failed to convert values: {e}", None, None

    X = np.array([values], dtype=float)
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0,1]) if hasattr(model, "predict_proba") else None

    if pred == 1:
        label_text = "Warning: The machine is prone to failure."
        suggestion = ("The result indicates a possible failure. I recommend having the machine checked immediately. "
                      "Checking the temperature, rotation speed, and instrument condition.")
    else:
        label_text = "The machine's condition appears stable now."
        suggestion = "The process is within normal values ​​— monitor performance periodically."
    prob_text = f"Potential for failure:{prob*100:.1f}%" if prob is not None else "The possibility is not available."
    human_msg = f"{label_text}\n{prob_text}\n\nrecommendation: {suggestion}"
    
    return human_msg, pred, round(prob, 4) if prob is not None else None

def build_ui():
    inputs = [gr.Textbox(label=col, placeholder="Enter a numerical value") for col in FEATURES_ORDER]
    output_text = gr.Textbox(label="Result (Message)", interactive=False)
    output_pred = gr.Number(label="prediction (0=no failure, 1=failure)")
    output_prob = gr.Number(label="probability (0..1)")

    demo = gr.Interface(
        fn=lambda *args: predict_human(**{FEATURES_ORDER[i]: args[i] for i in range(len(FEATURES_ORDER))}),
        inputs=inputs,
        outputs=[output_text, output_pred, output_prob],
        title="Predictive Maintenance — Fault Diagnostic Interface",
        description=("A simple interface for entering sensor readings and receiving a human recommendation.\n"
                     "Enter the values ​​in the same order as the fields or according to the labels shown."),
        examples=[
            [ "1", "300", "310", "1400", "40", "150", "0", "0", "1", "0", "0" ]
        ],
        allow_flagging="never",
        enable_queue=True
    )
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch()

