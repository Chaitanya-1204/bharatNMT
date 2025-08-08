import torch 
from model import get_model
import sentencepiece as spm

input = ["<2mr> તે સમયે રૃપાલીબેન ઘરમાં એકલા હતા." , 
         "<2en> તે સમયે રૃપાલીબેન ઘરમાં એકલા હતા." , 
         "<2hi> તે સમયે રૃપાલીબેન ઘરમાં એકલા હતા."]

TOKENIZER_MODEL_PATH = "tokenizer/spm_bharatnmt.model"
tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL_PATH)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = get_model()
state_dict = torch.load("best_model.pt", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Tokenize input

for sentence in input:
    src_ids = tokenizer.encode(sentence, out_type=int)
    src_ids = src_ids[:256]
    PAD_ID = tokenizer.pad_id()
    if len(src_ids) < 256:
        src_ids += [PAD_ID] * (256 - len(src_ids))
        
    print(f"Input: {sentence}")
    

    # Convert to tensor and move to device
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
    print(f"Output: {decoded}")
    print("=" * 80)
    