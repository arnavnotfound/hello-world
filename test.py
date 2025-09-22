from typing import Dict, List, Tuple
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import spacy

class RCAQualityMetrics:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def get_bert_embeddings(self, text: str) -> np.ndarray:
        """Generate BERT embeddings for given text."""
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using BERT embeddings."""
        emb1 = self.get_bert_embeddings(text1)
        emb2 = self.get_bert_embeddings(text2)
        return float(cosine_similarity(emb1, emb2)[0][0])

    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score between reference and candidate text."""
        reference_tokens = [word_tokenize(reference.lower())]
        candidate_tokens = word_tokenize(candidate.lower())
        return sentence_bleu(reference_tokens, candidate_tokens)

    def calculate_rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        scores = self.rouge_scorer.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    def extract_key_entities(self, text: str) -> List[str]:
        """Extract key entities and technical terms."""
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append(ent.text)
        return entities

    def entity_coverage_score(self, reference: str, candidate: str) -> float:
        """Calculate how well the candidate covers key entities from reference."""
        reference_entities = set(self.extract_key_entities(reference))
        candidate_entities = set(self.extract_key_entities(candidate))
        
        if not reference_entities:
            return 0.0
            
        return len(reference_entities.intersection(candidate_entities)) / len(reference_entities)

    def evaluate_rca(self, reference_rca: str, model_rca: str) -> Dict[str, float]:
        """
        Evaluate the quality of model's RCA against reference RCA.
        Returns a dictionary of different quality metrics.
        """
        metrics = {
            'semantic_similarity': self.calculate_semantic_similarity(reference_rca, model_rca),
            'bleu_score': self.calculate_bleu_score(reference_rca, model_rca),
            'entity_coverage': self.entity_coverage_score(reference_rca, model_rca)
        }
        
        # Add ROUGE scores
        rouge_scores = self.calculate_rouge_scores(reference_rca, model_rca)
        metrics.update(rouge_scores)
        
        # Calculate overall quality score (weighted average)
        weights = {
            'semantic_similarity': 0.3,
            'bleu_score': 0.2,
            'entity_coverage': 0.2,
            'rouge1': 0.1,
            'rouge2': 0.1,
            'rougeL': 0.1
        }
        
        metrics['overall_quality'] = sum(metrics[k] * weights[k] for k in weights.keys())
        
        return metrics

# Example usage
if __name__ == "__main__":
    evaluator = RCAQualityMetrics()
    
    reference_rca = """The application crashed due to memory overflow in the data processing module. 
    Root cause was identified as unbounded array growth in process_data() function."""
    
    model_rca = """System failure occurred because of memory issues in the data processor. 
    The process_data() function allowed unlimited array expansion causing memory overflow."""
    
    results = evaluator.evaluate_rca(reference_rca, model_rca)
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")