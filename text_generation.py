from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2-medium')
set_seed(42)

Prompts  = ['I am here to write amazon product review, this product is ', "Amazon product review, I am extremely disappointed with this product because it is terrible"]

for input in Prompts:
    generator(input, max_length=100,no_repeat_ngram_size=2,num_return_sequences=10,top_p=0.7)

