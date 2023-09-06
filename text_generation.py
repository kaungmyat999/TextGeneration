# -*- coding: utf-8 -*-
"""Text_Generation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QeY0DmW57vWf6Hdub-OS41uFDoJDuM7r
"""

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# pip install tensorflow transformers tensorflow_datasets

from transformers import pipeline,set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
output_text = generator("Hello, I'm a language model, ",max_length=300, num_return_sequences=1)
print(output_text)

output_text = str(output_text).split("<SEP>")[-1].strip()

generator("Write a poem about cloud ",max_length=50, num_return_sequences=3)

from transformers import pipeline

text2text_generator = pipeline("text2text-generation",model='t5-base')
#text2text_generator("question: What is 42 ? context: 42 is the answer to life, the universe and everything")

text2text_generator("translate from English to French: I'm very happy")

request = "text: Write a poem in English about the beauty of nature in spring."

# Generate the poem
output = text2text_generator(request, max_length=100, num_return_sequences=1)

# Extract the generated poem
poem = output[0]['generated_text']

# Print the generated poem
print("Generated Poem:")
print(poem)

input_text = "question: Do you think I am pretty?"

output = text2text_generator(input_text)

print(output)

text2text_generator2 = pipeline("text2text-generation",model='facebook/bart-base')
text2text_generator2(("UN Chief Says There Is No <mask> in Syria"))

request = "text: Write a review for the new XYZ smartphone. Highlight its camera quality, battery life, and overall performance."

# Generate the product review
output = text2text_generator2(request, max_length=150, num_return_sequences=1)

# Extract and print the generated product review
review = output[0]['generated_text']

# Print the generated review
print("Generated Product Review:")
print(review)

ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)
output = text2text_generator2([ARTICLE_TO_SUMMARIZE], max_length=150, num_return_sequences=1)
review = output[0]['generated_text']

from transformers import AutoTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

# Generate Summary
summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

from transformers import AutoTokenizer, BartForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("valhalla/bart-large-finetuned-squadv1")
model = BartForQuestionAnswering.from_pretrained("valhalla/bart-large-finetuned-squadv1")

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

# target is "nice puppet"
target_start_index = torch.tensor([14])
target_end_index = torch.tensor([15])
print(outputs)
outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)

import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxBartForConditionalGeneration

model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

text = "My friends are cool but they eat too many carbs."
inputs = tokenizer(text, max_length=1024, return_tensors="jax")
encoder_outputs = model.encode(**inputs)

decoder_start_token_id = model.config.decoder_start_token_id
decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

outputs = model.decode(decoder_input_ids, encoder_outputs)
last_decoder_hidden_states = outputs.hidden_states

last_decoder_hidden_states = outputs.hidden_states
print(last_decoder_hidden_states)

from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2-medium')
set_seed(42)
generator("I am here to write amazon product review, this product is ", max_length=70,num_return_sequences=3)

generator("I am here to write amazon product review, this product is ", max_length=100,no_repeat_ngram_size=2,num_return_sequences=10,top_p=0.7)

generated_text ={'generated_text': 'I am here to write amazon product review, this product is ive tested, love it.\n\nPros:\n It has the best price, it is fast, and it delivers the right price. Pros: It contains no chemicals, no preservatives and little taste. It is small, easy to carry and carries well. Cons: The'}

while not generated_text['generated_text'].endswith((".", "!", "?")):
    new_text = generator("", max_length=10, num_return_sequences=1)[0]['generated_text']
    generated_text['generated_text'] += new_text
    print(generated_text['generated_text'])

# Print the generated text
print("Generated Text:")
print(generated_text)

generator("I am here to write amazon product review, I don't like this  ", max_length=100,no_repeat_ngram_size=2,num_return_sequences=10,top_p=0.7)

