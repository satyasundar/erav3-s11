import re
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set

class OdiaBPETokenizer:
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        
        # Initialize basic Odia character vocabulary
        self.base_vocab = set()
        # Add basic Odia characters (vowels, consonants, marks)
        self._initialize_base_vocab()

    def _initialize_base_vocab(self):
        """Initialize vocabulary with basic Odia characters"""
        # Vowels
        self.base_vocab.update([chr(c) for c in range(0x0B05, 0x0B14)])
        # Consonants
        self.base_vocab.update([chr(c) for c in range(0x0B15, 0x0B39)])
        # Vowel marks
        self.base_vocab.update([chr(c) for c in range(0x0B3E, 0x0B4C)])
        # Other marks
        self.base_vocab.update([
            'ଂ', 'ଃ', 'ଁ', '୍',  # Anusvara, Visarga, Candrabindu, Halanta
            ' ', '\n', '\t'  # Whitespace characters
        ])

    def _get_stats(self, words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent pairs in the vocabulary"""
        pairs = defaultdict(int)
        for word in words:
            for i in range(len(word) - 1):
                pairs[tuple(word[i:i + 2])] += 1
        return pairs

    def _merge_vocab(self, words: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
        """Merge all occurrences of the most frequent pair"""
        first, second = pair
        new_words = []
        
        for word in words:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        
        return new_words

    def train(self, texts: List[str], min_freq: int = 2) -> None:
        """Train BPE model on texts"""
        # Split texts into characters
        words = []
        for text in texts:
            chars = list(text)
            # Filter valid Odia characters
            valid_chars = [c for c in chars if c in self.base_vocab or c.isspace()]
            if valid_chars:
                words.append(valid_chars)

        vocab = self.base_vocab.copy()
        num_merges = self.vocab_size - len(self.special_tokens) - len(vocab)
        print("num_merges : ", num_merges)
        print("total char length: ", len(words))
        # Perform BPE merges
        for i in range(num_merges):
            pairs = self._get_stats(words)
            if not pairs:
                break

            # Find most frequent pair
            best_pair = max(pairs.items(), key=lambda x: x[1])
            if best_pair[1] < min_freq:
                break

            pair = best_pair[0]
            new_token = ''.join(pair)
            vocab.add(new_token)
            print("merging ..", pair)
            # Record the merge operation
            self.merges[pair] = new_token
            
            # Merge the pair in all words
            words = self._merge_vocab(words, pair)

        # Build final vocabulary
        self.vocab = {**self.special_tokens}
        idx = len(self.special_tokens)
        for token in sorted(vocab):
            self.vocab[token] = idx
            idx += 1

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        """Encode text using learned BPE merges"""
        words = [list(text)]
        
        # Apply merges in order
        for pair, merged in self.merges.items():
            words = self._merge_vocab(words, pair)
        
        # Convert to token IDs
        result = []
        for word in words:
            for token in word:
                if token in self.vocab:
                    result.append(self.vocab[token])
                else:
                    result.append(self.special_tokens['<UNK>'])
        
        return result

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text"""
        return ''.join(self.inverse_vocab.get(id, '<UNK>') for id in ids)

    def calculate_compression_ratio(self, text: str) -> float:
        """Calculate compression ratio"""
        encoded = self.encode(text)
        return len(text) / len(encoded)

    def save(self, path: str) -> None:
        """Save tokenizer state"""
        # Convert tuple keys to strings for JSON serialization
        serializable_merges = {f"{first}|{second}": merged 
                              for (first, second), merged in self.merges.items()}
        
        data = {
            'vocab': self.vocab,
            'merges': serializable_merges,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'OdiaBPETokenizer':
        """Load tokenizer from file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.vocab = data['vocab']
        
        # Convert string keys back to tuples
        tokenizer.merges = {tuple(k.split('|')): v 
                           for k, v in data['merges'].items()}
        
        tokenizer.special_tokens = data['special_tokens']
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        return tokenizer 