from transformers import MBartConfig, MBartForConditionalGeneration
from logger import log as log_message


def get_model(config):
    
    model_config = MBartConfig(
        
        vocab_size = config["model"]["vocab_size"],
        max_position_embeddings = config["model"]["max_position_embeddings"],
        encoder_layers = config["model"]["encoder_layers"],
        decoder_layers = config["model"]["decoder_layers"],
        encoder_attention_heads = config["model"]["encoder_attention_heads"],
        decoder_attention_heads = config["model"]["decoder_attention_heads"],
        d_model = config["model"]["d_model"],
        encoder_ffn_dim = config["model"]["encoder_ffn_dim"],
        decoder_ffn_dim = config["model"]["decoder_ffn_dim"],
        dropout = config["model"]["dropout"],
        attention_dropout = config["model"]["attention_dropout"],
        activation_dropout = config["model"]["activation_dropout"],
        activation_function = config["model"]["activation_function"],
        pad_token_id = config["model"]["pad_token_id"],
        bos_token_id = config["model"]["bos_token_id"],
        eos_token_id = config["model"]["eos_token_id"],
        forced_bos_token_id = config["model"]["forced_bos_token_id"],
        scale_embedding = config["model"]["scale_embedding"],
        use_cache = config["model"]["use_cache"]
    )


    model = MBartForConditionalGeneration(model_config)
    log_message(f"Model loaded.... ", prefix="MODEL")
    
    return model

# Check parameter count
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    log_message(f"Model parameters: {trainable:,} trainable / {total:,} total" , prefix= "MODEL")
    
    


