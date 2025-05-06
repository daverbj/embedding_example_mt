
import torch
import json
import os
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    """Mean pooling operation to get sentence embeddings"""
    
    token_embeddings = model_output[0]
    
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class SentenceTransformerWrapper(torch.nn.Module):
    """Wrapper for sentence-transformer model to use in TorchScript"""
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = self.model.config
        
    def forward(self, input_ids, attention_mask):
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        embeddings = mean_pooling(outputs, attention_mask)
        
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
        
    def save_for_inference(self, output_path):
        """Save the model for C++ inference"""
        
        self.model.eval()
        
        
        
        sample_text = ""
        encoded = self.tokenizer(sample_text, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        
        
        with torch.no_grad():
            traced_model = torch.jit.trace(
                self,
                (input_ids, attention_mask)
            )
        
        
        traced_model.save(output_path)
        print(f"Model saved to {output_path}")
        
        
        tokenizer_config = {
            "vocab": self.tokenizer.get_vocab(),
            "special_tokens": {
                "pad_token": self.tokenizer.pad_token,
                "unk_token": self.tokenizer.unk_token,
                "cls_token": self.tokenizer.cls_token,
                "sep_token": self.tokenizer.sep_token
            },
            "model_max_length": self.tokenizer.model_max_length
        }
        
        
        config_path = os.path.join(os.path.dirname(output_path), "tokenizer_config.json")
        with open(config_path, 'w') as f:
            json.dump(tokenizer_config, f, indent=2)
        print(f"Tokenizer configuration saved to {config_path}")

def main():
    """Main function to export the model"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    output_dir = "."
    output_path = os.path.join(output_dir, "embeddings_model.pt")
    
    print(f"Exporting model: {model_name}")
    
    
    wrapper = SentenceTransformerWrapper(model_name)
    
    
    wrapper.save_for_inference(output_path)
    
    print("\nExport completed successfully!")
    print(f"Model saved at: {output_path}")
    print(f"You can now use this model for embedding your own strings.")

if __name__ == "__main__":
    main()
