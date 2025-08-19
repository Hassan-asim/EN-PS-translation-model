Usage Instructions:

1. *Install Dependencies*:
bash
pip install -r requirements.txt


2. *Prepare Your Dataset*:
   - Create a CSV file with two columns: Pashto text and English text
   - Update the csv_file parameter in the script

3. *Train the Model*:
bash
python translation_model.py


4. *Use Trained Model*: for testing use the below script 

.
.
.
# import the model 
from translation_model import load_model, translate_text

# Load trained model
model, tokenizer_en_to_ps, tokenizer_ps_to_en = load_model("./trained_model")

# Translate English to Pashto
english_text = "Hello, how are you?"
pashto_translation = translate_text(english_text, model.model_en_to_ps, tokenizer_en_to_ps)
print(f"Pashto: {pashto_translation}")

# Translate Pashto to English
pashto_text = "څه ورځ شه، تاسو څنګه یاست؟"
english_translation = translate_text(pashto_text, model.model_ps_to_en, tokenizer_ps_to_en)
print(f"English: {english_translation}")
"# EN-PS-translation-model"  
