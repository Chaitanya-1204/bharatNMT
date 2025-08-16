# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from model import get_model
import sentencepiece as spm
import yaml
import os 

app = Flask(__name__)

CORS(app)

TOKENIZER_MODEL_PATH = "tokenizer/spm_bharatnmt.model"
tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL_PATH)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


model = get_model(config)
state_dict = torch.load("models/best_model.pt", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

def translate(final_text):
    src_ids = tokenizer.encode(final_text, out_type=int)
    src_ids = src_ids[:256]
    PAD_ID = tokenizer.pad_id()
    if len(src_ids) < 256:
        src_ids += [PAD_ID] * (256 - len(src_ids))
        
    input_ids = torch.tensor([src_ids], dtype=torch.long).to(device)
    attention_mask = (input_ids != tokenizer.pad_id()).long().to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256,
            num_beams=4,
            early_stopping=True,
            bos_token_id=tokenizer.bos_id(),
            eos_token_id=tokenizer.eos_id()
        )

    # Decode output
    decoded = tokenizer.decode(output_ids[0].tolist())
    
    return decoded


@app.route('/translate', methods=['POST'])
def translate_text():
    

    
    data = request.get_json()
    text = data.get('text')
    source_lang = data.get('source_lang')
    
    target_lang = data.get('target_lang')
    print(f"Translating: '{text}' from {source_lang} to {target_lang}")
    final_text = ""
    if(source_lang == "en") :
        tl = "<2" + target_lang + ">"
        
        final_text = tl + " " + text
        
    else:
        
        if(target_lang == "en"):
            tl = "<2" + target_lang + ">"
            final_text = tl + " " + text
            
        
        
        else:
            tl = "<2en>"
            final_text = tl + " " + text 
            
            decoded_text = translate(final_text)
            
            tl = "<2" + target_lang + "> "
            final_text = tl + decoded_text
    # Translation 
    
    
    decoded = translate(final_text)
    

    return jsonify({"translated_text":decoded})

    

# --- 6. Run the Server ---
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)


