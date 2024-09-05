from torchmetrics.functional.multimodal.clip_score import clip_score
from functools import partial
import torch
from PIL import Image
import numpy as np

class CLIPLoss(torch.nn.Module):

    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.model = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    def forward(self, image, text):

        clip_score = self.model(image,
                                text)
        similarity = 1 - clip_score / 100

        return similarity

#clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

if __name__ == "__main__":

    cliploss = CLIPLoss()
    ours_path = "/nfs/data_chaos/jzhang/dataset/test_clip/43.jpg"
    image = Image.open(ours_path)
    image = np.array(image)
    prompt = "a man with black hair, without beard"
    sd_clip_score = cliploss(image, prompt).cpu().detach().numpy()
    print("CLIP score:", sd_clip_score)

# ####################################################################################################
