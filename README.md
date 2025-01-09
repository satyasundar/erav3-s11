# Odia Tokenizer

Here we are going to build a Odia Tokennizer for inputing text into LLMs.

## Base Vocabulary
-  List all the unicode characters for Odia Language
-  Include all the vowels, consonants, signs, dependent vowels in the list
-  Make this whole list as base vocabulary

## Preprocess
- Collect the odia corpus data from huggingface or wikipedia
- I have taken dump from huggingface datasets for training. But during development I was using odia wikipedia data
- Make a regex pattern to separate odia words including spaces
    ```
    r""" ?[\u0B00-\u0B7F]+| ?[^\s]+|\s+(?!\S)|\s+"""
    ```
- Then input this word to train method using Byte-Pair-Encoding algorithm

## Train
- This method will take the separated words from preprocess step.
- It will pair the characters using BPE
- It will use the get_stats method to pair and find the maximum occuring pairs.
- After finding the most common pair, it will call the merge method to merge the pair with a new token.
- Finally vocabulary gets updated.

## Inference
- It has got encode method which can take the input text and provide the BPE codes
- It has also got decode method which will show the inputted text from BPE codes

## Results
- It has achieved compression ratio of more than 3.2 in many input text.
- It can show the same compression for new data as well.
- But It fails to achive this compression ratio in some cases as well.
- It has resullted 5000 as output tokens consisting the new vocabulary.

## Deployment
- This is deployed in HUggingface Spaces. The link is provided below.

    https://huggingface.co/spaces/satyanayak/odia_tokenizer_5k

- This can generate the BPE code as we type into input area dynamically