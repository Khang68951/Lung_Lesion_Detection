---
library_name: transformers
pipeline_tag: image-feature-extraction
license: apache-2.0
tags:
- dino
- vision
inference: false
---

# Vision Transformer (small-sized model) trained using DINOv2, with registers

Vision Transformer (ViT) model introduced in the paper [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) by Darcet et al. and first released in [this repository](https://github.com/facebookresearch/dinov2). 

Disclaimer: The team releasing DINOv2 with registers did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

The Vision Transformer (ViT) is a transformer encoder model (BERT-like) [originally introduced](https://arxiv.org/abs/2010.11929) to do supervised image classification on ImageNet.

Next, people figured out ways to make ViT work really well on self-supervised image feature extraction (i.e. learning meaningful features, also called embeddings) on
images without requiring any labels. Some example papers here include [DINOv2](https://huggingface.co/papers/2304.07193) and [MAE](https://arxiv.org/abs/2111.06377).

The authors of DINOv2 noticed that ViTs have artifacts in attention maps. It’s due to the model using some image patches as “registers”. The authors propose a fix: just add some new tokens (called "register" tokens), which you only use during pre-training (and throw away afterwards). This results in:
- no artifacts
- interpretable attention maps
- and improved performances.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dinov2_with_registers_visualization.png"
alt="drawing" width="600"/>

<small> Visualization of attention maps of various models trained with vs. without registers. Taken from the <a href="https://arxiv.org/abs/2309.16588">original paper</a>. </small>

Note that this model does not include any fine-tuned heads. 

By pre-training the model, it learns an inner representation of images that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled images for instance, you can train a standard classifier by placing a linear layer on top of the pre-trained encoder. One typically places a linear layer on top of the [CLS] token, as the last hidden state of this token can be seen as a representation of an entire image.

## Intended uses & limitations

You can use the raw model for feature extraction. See the [model hub](https://huggingface.co/models?other=dinov2_with_registers) to look for
fine-tuned versions on a task that interests you.

### How to use

Here is how to use this model:

```python
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-with-registers-small')
model = AutoModel.from_pretrained('facebook/dinov2-with-registers-small')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
```

### BibTeX entry and citation info

```bibtex
@misc{darcet2024visiontransformersneedregisters,
      title={Vision Transformers Need Registers}, 
      author={Timothée Darcet and Maxime Oquab and Julien Mairal and Piotr Bojanowski},
      year={2024},
      eprint={2309.16588},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2309.16588}, 
}
```