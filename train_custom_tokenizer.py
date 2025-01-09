from odia_tokenizer import OdiaBPETokenizer

def load_odia_texts(file_paths):
    texts = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return texts

# Training
data_files = [
    "../A11_Odia_Tokenizer/odia_corpus_clean.txt",
]

# Create and train tokenizer
tokenizer = OdiaBPETokenizer(vocab_size=5000)
texts = load_odia_texts(data_files)

texts[0] = texts[0][:2000000]


tokenizer.train(texts, min_freq=10)

# Test the tokenizer
test_text = "ଏହା ବହୁତ ସାଧାରଣ ହୋଇଗଲାଣି । ଏହା ପରେ ମଧ୍ୟ ସେ ବିଜ୍ଞାନ ଜଗତରେ ନିଜ ଗବେଷଣା ଜାରି ରଖିଥିଲେ ।"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)

# Calculate compression ratio
compression_ratio = tokenizer.calculate_compression_ratio(test_text)

print(f"Original text: {test_text}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
print(f"Compression ratio: {compression_ratio:.2f}")

# Print some statistics
print(f"\nVocabulary size: {len(tokenizer.vocab)}")
print(f"Number of merges: {len(tokenizer.merges)}")
print("\nSample merges:")
for i, (pair, merged) in enumerate(list(tokenizer.merges.items())[:10]):
    print(f"{pair} -> {merged}")

# Save the tokenizer
tokenizer.save("odia_bpe_tokenizer.json") 