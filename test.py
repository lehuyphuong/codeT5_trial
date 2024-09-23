import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the fine-tuned model and tokenizer
# Directory where your fine-tuned model is saved
model_dir = "D:/2024/Projects/C2F/Appoarch_1/CodeT5/Analysis_room/CP/fine-tuned-codeT5"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# Input C code
c_code = """
int add(int a, int b) {
    return a + b;
}
"""

# Tokenize the input C code
input_ids = tokenizer.encode(c_code, return_tensors="pt")

# Generate the output (JavaScript code) from the model
model.eval()
with torch.no_grad():
    output_ids = model.generate(
        input_ids, max_length=100, num_beams=5, early_stopping=True)

# Decode the generated output into JavaScript code
js_code = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print the output JS code
print("Input C code:")
print(c_code)
print("\nGenerated JavaScript code:")
print(js_code)
