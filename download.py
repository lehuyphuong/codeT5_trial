from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Specify the model checkpoint
model_name = "Salesforce/codet5-base"

# Download the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Optionally, save the model and tokenizer to a local directory
model.save_pretrained("./codet5-base")
tokenizer.save_pretrained("./codet5-base")

print("Model and tokenizer downloaded successfully!")