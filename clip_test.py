from PIL import Image
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import onnxruntime


class ImageEncoder(nn.Module):
    def __init__(self, clip_model: CLIPModel, input_ids=None):
        super().__init__()
        self.input_ids = input_ids
        self.vision_model = clip_model.vision_model
        self.text_model = clip_model.text_model
        self.visual_projection = clip_model.visual_projection
        self.text_projection = clip_model.text_projection
        self.logit_scale = clip_model.logit_scale

    def forward(self, pixel_values):
        image_embedding = self.vision_model(pixel_values)[1]
        image_embedding = self.visual_projection(image_embedding)
        if self.input_ids is None:
            return image_embedding
        else:
            text_embedding = self.text_model(self.input_ids)[1]
            text_embedding = self.text_projection(text_embedding)

            # normalized features
            image_embedding = image_embedding / image_embedding.norm(p=2, dim=-1, keepdim=True)
            text_embedding = text_embedding / text_embedding.norm(p=2, dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits_per_text = torch.matmul(text_embedding, image_embedding.t()) * logit_scale
            return logits_per_text.t()


model = CLIPModel.from_pretrained("./models/openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("./models/openai/clip-vit-base-patch32")


image = Image.open("./CLIP.png")
texts = ["a diagram", "a dog", "a cat"]

image_encoder = ImageEncoder(model)
image_encoder.eval()
# text_encoder = model.text_model

input_ = processor(text=texts, images=image, return_tensors="pt")
# pixel_values = input_.pixel_values
pixel_values = torch.randn(1, 3, 224, 224)
input_ids = input_.input_ids
attention_mask = input_.attention_mask

# 直接计算结果


# # 导出image_onnx模型
# torch.onnx.export(
#     image_encoder,
#     pixel_values,
#     "./onnx/image_encoder.onnx",
#     opset_version=13,
#     do_constant_folding=True,
#     input_names=["pixel_values"],
#     output_names=["image_embeddingfsa"],
# )

# image_encoder_with_texts = ImageEncoder(model, input_ids)

# # 导出image_onnx_with_texts模型
# torch.onnx.export(
#     image_encoder_with_texts,
#     pixel_values,
#     "./onnx/image_onnx_with_text.onnx",
#     opset_version=13,
#     do_constant_folding=True,
#     input_names=["pixel_values"],
#     output_names=["logit"],
#     dynamic_axes={"pixel_values": {0: "batch_size"}, "logit": {0: "batch_size"}},
# )
