import gradio as gr
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "mai-data-ai/emotion-distilbert"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
labels = [mdl.config.id2label[i] for i in range(mdl.config.num_labels)]

EMOJI = {"joy": "😄", "sadness": "😢", "anger": "😠",
         "fear": "😨", "surprise": "😲", "disgust": "🤢"}


def predict(text):
    if not text or not text.strip():
        return "Please enter some text.", None
    enc = tok(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        probs = torch.softmax(mdl(**enc).logits, dim=-1)[0].tolist()
    top = max(range(len(probs)), key=probs.__getitem__)
    df = pd.DataFrame({"emotion": labels, "probability": probs})
    return f"{EMOJI[labels[top]]} {labels[top].upper()} ({probs[top]:.0%})", df


with gr.Blocks(title="Emotion Classifier") as demo:
    gr.Markdown("# 🎭 Emotion Classifier\n"
                "Type a sentence and see which of 6 emotions it expresses "
                "(joy, sadness, anger, fear, surprise, disgust).")
    inp = gr.Textbox(label="Your text", lines=3,
                     placeholder="e.g. I can't believe how amazing this is!")
    btn = gr.Button("Analyze", variant="primary")
    out_label = gr.Textbox(label="Dominant emotion")
    out_plot = gr.BarPlot(x="emotion", y="probability", y_lim=[0, 1],
                          title="Confidence per emotion")
    gr.Examples(
        ["I am so happy and grateful for everything!",
         "I'm terrified, something is following me",
         "This is disgusting, I can't believe anyone would eat that",
         "I miss her so much, I feel so empty",
         "Wow, I did not expect that at all!",
         "Stop wasting my time, this is so annoying"],
        inputs=inp)
    btn.click(predict, inp, [out_label, out_plot])
    inp.submit(predict, inp, [out_label, out_plot])

if __name__ == "__main__":
    demo.launch()
