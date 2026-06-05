import os
import re
import json
from typing import Set, List
from rapidfuzz import process, fuzz
from app.core.config import settings
from app.core.logging import logger

class TypoHandler:
    def __init__(self):
        self.vocab: Set[str] = set()
        self._loaded = False

    def load_vocab(self) -> bool:
        if self._loaded:
            return True
        
        words_set = set()
        
        # Load from general chunks
        meta_path = settings.chunk_metadata_path
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                for item in metadata:
                    text = item.get("text", "")
                    words_set.update(re.findall(r"\b\w+\b", text.lower()))
            except Exception as e:
                logger.error(f"Error loading general metadata vocabulary: {e}")

        # Load from disease records
        disease_path = settings.disease_demo_path
        if os.path.exists(disease_path):
            try:
                with open(disease_path, "r", encoding="utf-8") as f:
                    diseases = json.load(f)
                for item in diseases:
                    words_set.update(re.findall(r"\b\w+\b", item.get("condition_name", "").lower()))
                    for sym in item.get("symptoms", []):
                        words_set.update(re.findall(r"\b\w+\b", sym.lower()))
                    words_set.update(re.findall(r"\b\w+\b", item.get("description", "").lower()))
            except Exception as e:
                logger.error(f"Error loading disease vocabulary: {e}")

        # Load from medicine records
        medicine_path = settings.medicine_demo_path
        if os.path.exists(medicine_path):
            try:
                with open(medicine_path, "r", encoding="utf-8") as f:
                    medicines = json.load(f)
                for item in medicines:
                    words_set.update(re.findall(r"\b\w+\b", item.get("medicine_name", "").lower()))
                    words_set.update(re.findall(r"\b\w+\b", item.get("generic_name", "").lower()))
                    for alias in item.get("aliases", []):
                        words_set.update(re.findall(r"\b\w+\b", alias.lower()))
                    words_set.update(re.findall(r"\b\w+\b", item.get("category", "").lower()))
                    for use in item.get("uses", []):
                        words_set.update(re.findall(r"\b\w+\b", use.lower()))
                    words_set.update(re.findall(r"\b\w+\b", item.get("description", "").lower()))
            except Exception as e:
                logger.error(f"Error loading medicine vocabulary: {e}")

        self.vocab = words_set
        self._loaded = True
        logger.info(f"TypoHandler loaded vocabulary with {len(self.vocab)} words.")
        return True


    def normalize_text(self, text: str) -> str:
        text = text.replace("\t", " ").replace("\r", "")
        text = re.sub(r"[^\w\s\-\.\,\/\d]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def correct_query(self, query: str) -> str:
        self.load_vocab()
        normalized = self.normalize_text(query)
        if not self.vocab:
            return normalized
            
        words = normalized.split()
        corrected_words = []
        
        for word in words:
            word_lower = word.lower()
            # If word is numeric, very short, or already matches a word in the vocab, preserve it
            if word_lower.isdigit() or len(word_lower) <= 3 or word_lower in self.vocab:
                corrected_words.append(word)
            else:
                # Find closest match in corpus vocabulary using fuzzy matching
                match = process.extractOne(
                    word_lower,
                    self.vocab,
                    scorer=fuzz.WRatio,
                    score_cutoff=85.0
                )
                if match:
                    corrected_words.append(match[0])
                else:
                    corrected_words.append(word)
                    
        return " ".join(corrected_words)
        
    def reset_vocab(self):
        self.vocab = set()
        self._loaded = False
