import gradio as gr
import json
from odia_tokenizer import OdiaBPETokenizer
import random
import colorsys

def generate_distinct_colors(n):
    """Generate n visually distinct colors"""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors

def load_tokenizer():
    try:
        return OdiaBPETokenizer.load("odia_bpe_tokenizer.json")
    except:
        # If no saved tokenizer found, create a new one
        return OdiaBPETokenizer(vocab_size=5000)

def tokenize_text(text):
    tokenizer = load_tokenizer()
    
    # Get token IDs and their corresponding text
    token_ids = tokenizer.encode(text)
    tokens = []
    current_pos = 0
    
    # Process text to get token spans
    words = [list(text)]
    for pair, merged in tokenizer.merges.items():
        words = tokenizer._merge_vocab(words, pair)
    
    # Extract final tokens
    final_tokens = []
    for word in words:
        final_tokens.extend(word)
    
    # Generate colors for tokens
    colors = generate_distinct_colors(len(tokenizer.vocab))
    color_map = {token_id: color for token_id, color in zip(tokenizer.vocab.values(), colors)}
    
    # Create highlighted HTML
    html_parts = []
    token_list = []
    
    for i, token in enumerate(final_tokens):
        token_id = tokenizer.vocab.get(token, tokenizer.special_tokens['<UNK>'])
        color = color_map[token_id]
        html_parts.append(f'<span style="background-color: {color}">{token}</span>')
        token_list.append(f"{token} ({token_id})")
    
    highlighted_text = "".join(html_parts)

    # Calculate compression ratio
    compression_ratio = len(text) / len(token_ids) if len(token_ids) > 0 else 0
    
    return (
        len(token_ids),  # Token count
        compression_ratio,  # Compression ratio
        highlighted_text,  # Highlighted text
        "\n".join(token_list)  # Token list
    )

custom_css = """
.token-highlight {
    border-radius: 3px;
    margin: 0 1px;
}
.container {
    max-width: 1200px;
    margin: 0 auto;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Odia BPE Tokenizer")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Input Text",
                placeholder="Enter Odia text here...",
                lines=10
            )
        
        with gr.Column(scale=1):
            token_count = gr.Number(label="Token Count")
            compression_ratio = gr.Number(label="Compression Ratio")
            highlighted_output = gr.HTML(label="Tokenized Text")
            token_list = gr.Textbox(label="Token List", lines=10)
    
    input_text.change(
        fn=tokenize_text,
        inputs=[input_text],
        outputs=[token_count, compression_ratio, highlighted_output, token_list]
    )

demo.launch() 