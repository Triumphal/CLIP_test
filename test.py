from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from optimum.exporters.onnx import main_export

# model_checkpoint = "distilbert_base_uncased_squad"
model_path = "./models/openai/clip-vit-base-patch32"
# save_directory = "onnx/"
#
# # Load a model from transformers and export it to ONNX
# ort_model = ORTModelForSequenceClassification.from_pretrained(model_checkpoint, export=True)
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#
# # Save the onnx model and tokenizer
# ort_model.save_pretrained(save_directory)
# tokenizer.save_pretrained(save_directory)

main_export(model_name_or_path=model_path,
            task="feature-extraction",
            output="./models/openai/clip-vit-base-patch32_onnx/")
