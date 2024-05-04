# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# class MedicalQA:
#     def __init__(self):
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         model_path = '../gpt2-medquad-finetuned-20240428'
#         self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
#          # Quantize the model to improve inference time
#         # self.model = torch.quantization.quantize_dynamic(
#         #     self.model, {torch.nn.Linear}, dtype=torch.qint8
#         # )
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.pipeline = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, device=0 if self.device == 'cuda' else -1)
#         # self.history = []

#         # Define predefined responses
#         self.predefined_responses = {
#             "hi": "Hello! How can I assist you today?",
#             "hello": "Hello there! What can I do for you?",
#             "thanks": "You're welcome! Anything else I can help with?",
#             "thank you": "You're welcome! Feel free to ask any more questions.",
#             "sorry": "No problem at all! How can I assist you further?",
#             "help": "Sure! Please ask any medical question you have in mind.",
#             "what can you do": "I can assist you with medical questions and health advice. What do you need help with today?"
#         }

#     def ask(self, question, temperature=0.7, top_p=0.85, repetition_penalty=1.1, do_sample=False):
#         # Check if the question is a predefined query
#         normalized_question = question.lower().strip()
#         if normalized_question in self.predefined_responses:
#             return self.predefined_responses[normalized_question]

#         # prompt = f"Last Info: {self.history[-1] if self.history else ''}\nNew Question: {question}\nAnswer:"
#         prompt = f"Question: {question} Answer:"
#         response = self.pipeline(prompt, max_length=50, pad_token_id=self.tokenizer.eos_token_id, temperature=temperature,
#                                  top_p=top_p, repetition_penalty=repetition_penalty, do_sample=do_sample)[0]['generated_text']
#         answer = response.split('Answer:')[1].split('.')[0] + '.' if 'Answer:' in response else "Sorry, I couldn't understand the question."
#         # self.history.append(answer)
#         return answer

import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, pipeline

# Check device availability and set the model to use GPU/CPU accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def load_model_and_tokenizer(model_path):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    return model, tokenizer

model_path = "../gpt2-medquad-finetuned"
model, tokenizer = load_model_and_tokenizer(model_path)

def prepare_input(tokenizer, input_text):
    prompt = f"Question: {input_text} Answer:"
    encoded_input = tokenizer.encode(prompt, return_tensors='pt')
    return encoded_input.to(device)  # Ensure the tensor is on the correct device

def generate_text(model, tokenizer, encoded_input):
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            encoded_input,
            max_length=512,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            temperature=1.0,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=False
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, pipeline


predefined_responses = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hello there! What can I do for you?",
    "thanks": "You're welcome! Anything else I can help with?",
    "thank you": "You're welcome! Feel free to ask any more questions.",
    "sorry": "No problem at all! How can I assist you further?",
    "help": "Sure! Please ask any medical question you have in mind.",
    "what can you do": "I can assist you with medical questions and health advice. What do you need help with today?"
}



def answer_query(input_text):
    normalized_text = input_text.strip().lower()
    if normalized_text in predefined_responses:
        return predefined_responses[normalized_text]
    
    encoded_input = prepare_input(tokenizer, input_text)
    try:
        return generate_text(model, tokenizer, encoded_input).split("Answer:")[1].strip()
    except:
        return "Apologize, I can understand this query. Would you please ask another one , please."


