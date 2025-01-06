import torch
import torch.nn.functional as F
from hierarchical_blt import HierarchicalBLT
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize
import argparse
from pathlib import Path

class HierarchicalGenerator:
    """
    Generator that uses hierarchical BLT model for controlled text generation
    with paragraph-level planning
    """
    def __init__(
        self,
        model_path,
        paragraph_model_name="facebook/bart-large-mnli",
        device=None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Load hierarchical BLT model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = HierarchicalBLT(
            d_model=512,
            n_layers=24,
            n_heads=8,
            encoder_layers=1,
            decoder_layers=9,
            window_size=512,
            max_ngram=8,
            hash_vocab_size=300000,
            dropout=0.1,
            paragraph_dim=1024
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load paragraph embedding model
        self.paragraph_tokenizer = AutoTokenizer.from_pretrained(paragraph_model_name)
        self.paragraph_model = AutoModelForSequenceClassification.from_pretrained(
            paragraph_model_name
        ).to(self.device)
        self.paragraph_model.eval()
    
    def get_paragraph_embedding(self, text):
        """Get semantic embedding for paragraph"""
        with torch.no_grad():
            inputs = self.paragraph_tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            outputs = self.paragraph_model(**inputs)
            return outputs.hidden_states[-1][:, 0].squeeze()
    
    def generate_with_plan(
        self,
        prompt,
        outline=None,
        max_length=2048,
        temperature=0.8,
        top_p=0.9,
        num_paragraphs=3
    ):
        """
        Generate text with paragraph-level planning
        Args:
            prompt: Initial text prompt
            outline: List of paragraph descriptions for planning
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            num_paragraphs: Number of paragraphs to generate if no outline
        """
        # Convert prompt to bytes
        prompt_bytes = torch.tensor(
            [b for b in prompt.encode()],
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # Get paragraph embeddings from outline or generate them
        if outline is not None:
            paragraph_embeddings = []
            for desc in outline:
                emb = self.get_paragraph_embedding(desc)
                paragraph_embeddings.append(emb)
            paragraph_embeddings = torch.stack(paragraph_embeddings)
        else:
            # Generate paragraph embeddings autoregressively
            paragraph_embeddings = []
            current_text = prompt
            
            for _ in range(num_paragraphs):
                # Get embedding for current text
                current_emb = self.get_paragraph_embedding(current_text)
                
                # Generate next paragraph embedding
                with torch.no_grad():
                    # Use model's paragraph encoder to predict next embedding
                    next_emb = self.model.paragraph_encoder(
                        current_emb.unsqueeze(0).unsqueeze(0),
                        torch.ones(1, 1, dtype=torch.bool, device=self.device)
                    )
                    paragraph_embeddings.append(next_emb.squeeze())
                
                # Generate text for this paragraph
                new_text = self.generate_paragraph(
                    current_text,
                    next_emb,
                    max_length=max_length // num_paragraphs,
                    temperature=temperature,
                    top_p=top_p
                )
                current_text = new_text
            
            paragraph_embeddings = torch.stack(paragraph_embeddings)
        
        # Generate full text with paragraph plan
        generated = self.model.generate(
            prompt_bytes,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            paragraph_plan=paragraph_embeddings
        )
        
        # Convert back to text
        try:
            text = bytes(generated[0].cpu().tolist()).decode('utf-8')
            return text
        except UnicodeDecodeError:
            # Handle invalid UTF-8 bytes by replacing them
            return bytes(generated[0].cpu().tolist()).decode('utf-8', errors='replace')
    
    def generate_paragraph(
        self,
        context,
        paragraph_embedding,
        max_length=512,
        temperature=0.8,
        top_p=0.9
    ):
        """Generate a single paragraph conditioned on embedding"""
        # Convert context to bytes
        context_bytes = torch.tensor(
            [b for b in context.encode()],
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # Generate with paragraph conditioning
        generated = self.model.generate(
            context_bytes,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            paragraph_plan=paragraph_embedding.unsqueeze(0)
        )
        
        # Convert back to text
        try:
            text = bytes(generated[0].cpu().tolist()).decode('utf-8')
            return text
        except UnicodeDecodeError:
            return bytes(generated[0].cpu().tolist()).decode('utf-8', errors='replace')

def main():
    parser = argparse.ArgumentParser(description='Generate text with hierarchical BLT model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--prompt', type=str, required=True, help='Initial text prompt')
    parser.add_argument('--outline', type=str, nargs='+', help='Optional paragraph descriptions')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling threshold')
    parser.add_argument('--num_paragraphs', type=int, default=3, help='Number of paragraphs if no outline')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = HierarchicalGenerator(args.model_path)
    
    # Generate text
    generated_text = generator.generate_with_plan(
        prompt=args.prompt,
        outline=args.outline,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        num_paragraphs=args.num_paragraphs
    )
    
    # Save or print output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(generated_text)
        print(f"Generated text saved to: {args.output}")
    else:
        print("\nGenerated Text:")
        print("=" * 80)
        print(generated_text)
        print("=" * 80)

if __name__ == "__main__":
    main()
