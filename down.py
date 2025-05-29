# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token="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")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token="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")

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token="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")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token="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") yao

# mistralai/Mistral-Small-3.1-24B-Instruct-2503  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

# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", token="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")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", token="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") xn