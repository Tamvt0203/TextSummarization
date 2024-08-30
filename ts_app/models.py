from django.db import models
from transformers import TFAutoModelForSeq2SeqLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

class ModelLoader:
    def __init__(self, model_checkpoints, weights_files):
        """
        model_checkpoints: dictionary containing model checkpoints for 'en' (TensorFlow) and 'vi' (PyTorch)
        weights_files: dictionary containing weights files for 'en' (TensorFlow) and 'vi' (PyTorch)
        """
        self.models = {}
        self.tokenizers = {}
        self.summarizers = {}
        
        # Load models for both languages
        self.models['en'] = self.load_torch_model(model_checkpoints['en'], weights_files['en'])
        self.models['vi'] = self.load_torch_model(model_checkpoints['vi'], weights_files['vi'])
        
        self.tokenizers['en'] = AutoTokenizer.from_pretrained(model_checkpoints['en'])
        self.tokenizers['vi'] = AutoTokenizer.from_pretrained(model_checkpoints['vi'])
        
        self.summarizers['en'] = self.create_pipeline('en', 'tf')
        self.summarizers['vi'] = self.create_pipeline('vi', 'pt')

    def load_tf_model(self, model_checkpoint, weights_file):
        """Load and return a TensorFlow model for a given checkpoint and weights."""
        model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        model.load_weights(weights_file)
        return model

    def load_torch_model(self, model_checkpoint, weights_file):
        """Load and return a PyTorch model for a given checkpoint and weights."""
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        model.load_state_dict(torch.load(weights_file))
        return model
    
    def create_pipeline(self, language, framework):
        """Create and return the summarization pipeline for a given language."""
        return pipeline("summarization", model=self.models[language], tokenizer=self.tokenizers[language], framework=framework)

    def summarize_text(self, text, language="en", min_length=5, max_length=128):
        """Generate summary for a given text in specified language."""
        if language in self.summarizers:
            summary = self.summarizers[language](text, min_length=min_length, max_length=max_length)
            return summary[0]['summary_text']
        else:
            raise ValueError("Unsupported language")
