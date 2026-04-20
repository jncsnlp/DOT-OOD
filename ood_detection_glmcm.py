import torch
import os
import numpy as np
import clip
from class_names import CLASS_NAME, prompt_templates
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from build_dataset import ImageTextDataset
from measure import auc, fpr_recall

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detection(text_features, dataloader, clip_model, dataset_name, lambd, t):
    scores = []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc=dataset_name):
            imgs = imgs.to(device)
            image_features_global, image_features_local = clip_model.encode_image(imgs)

            image_features_global = image_features_global.to(torch.float32)
            image_features_local = image_features_local.to(torch.float32)

            image_features_global = image_features_global / image_features_global.norm(dim=-1, keepdim=True)
            image_features_local = image_features_local / image_features_local.norm(dim=-1, keepdim=True)

            logits_global = image_features_global @ text_features.T
            prob_global = F.softmax(logits_global / t, dim=1).cpu().numpy()

            logits_local = image_features_local @ text_features.T
            prob_local = F.softmax(logits_local / t, dim=-1).cpu().numpy()

            global_score = np.max(prob_global, axis=1)
            local_score = np.max(prob_local, axis=(1, 2, 3))

            batch_score = global_score + lambd * local_score
            scores.append(batch_score)
        scores = np.concatenate(scores, axis=0)
    return scores


def glmcm(model_path="pretrain"):
    recall = 0.95
    batch_size = 256
    shuffle = True
    arch = "ViT-B/16"
    prompt_idx_pos = 39

    method = "GLMCM"

    with open(f"{model_path}/{method}.csv", "w") as fw:
        fw.write(f"{arch} {model_path}\n")

        clip_model, preprocess = clip.load(arch, device, jit=False)
        if not model_path == "pretrain":
            if os.path.exists(model_path):
                checkpoint = torch.load(f"{model_path}/checkpoint.pth", weights_only=False)
                clip_model.load_state_dict(checkpoint["model_state_dict"])

                print("Loaded the best model for final evaluation.")
        clip_model.eval()

        dataset_val = ImageTextDataset(
            f"data/imagenet/",
            f"datalists/imagenet2012_val.txt",
            preprocess,
        )
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle, num_workers=4)

        ood_names = ["inaturalist", "sun", "places", "texture"]
        dataset_oods = [
            ImageTextDataset(
                f"data/{ood_name}/",
                f"datalists/{ood_name}.txt",
                preprocess,
            )
            for ood_name in ood_names
        ]
        dataloader_oods = [
            DataLoader(dataset_ood, batch_size=batch_size, shuffle=shuffle, num_workers=4) for dataset_ood in dataset_oods
        ]

        train_dataset = "imagenet"
        class_name = CLASS_NAME[train_dataset]
        prompts = [prompt_templates[prompt_idx_pos].format(c) for c in class_name]
        text_inputs = torch.cat([clip.tokenize(f"{c}") for c in prompts]).to(device)

        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs).to(torch.float32)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        lambd = 0.5
        t = 1.0

        fw.write(f"method: {method}, lambd: {lambd}, t: {t}\n")
        scores_id = detection(text_features, dataloader_val, clip_model, "id_val", lambd, t)

        fpr_all = 0
        auroc_all = 0
        for ood_name, dataloader_ood in zip(ood_names, dataloader_oods):
            scores_ood = detection(text_features, dataloader_ood, clip_model, ood_name, lambd, t)

            auc_ood = auc(scores_id, scores_ood)[0]
            fpr_ood, _ = fpr_recall(scores_id, scores_ood, recall)
            fpr_all += fpr_ood
            auroc_all += auc_ood

            print(f"{ood_name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}")
            # break
        fw.write(f"mean auroc {auroc_all/len(ood_names):.2%}, mean fpr {fpr_all/len(ood_names):.2%}\n\n")


if __name__ == "__main__":
    glmcm()
