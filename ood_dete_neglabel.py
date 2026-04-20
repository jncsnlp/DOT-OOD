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


def detection(text_features_pos, text_features_neg, dataloader, clip_model, dataset_name, ngroup, random_permute=False):
    all_scores = []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc=dataset_name):
            imgs = imgs.to(device)
            image_features, _ = clip_model.encode_image(imgs)
            image_features = image_features.to(torch.float32)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            pos_sim = 100.0 * image_features @ text_features_pos.T
            neg_sim = 100.0 * image_features @ text_features_neg.T

            drop = neg_sim.shape[1] % ngroup
            if drop > 0:
                neg_sim = neg_sim[:, :-drop]

            if random_permute:
                SEED = 0
                torch.manual_seed(SEED)
                torch.cuda.manual_seed(SEED)
                idx = torch.randperm(neg_sim.shape[1], device=device)
                neg_sim = neg_sim.T
                negs_sim = neg_sim[idx].T.reshape(pos_sim.shape[0], ngroup, -1).contiguous()
            else:
                negs_sim = neg_sim.reshape(pos_sim.shape[0], ngroup, -1).contiguous()

            batch_scores = []
            for j in range(ngroup):
                full_sim = torch.cat([pos_sim, negs_sim[:, j, :]], dim=-1)
                full_sim = full_sim.softmax(dim=-1)
                pos_score = full_sim[:, : pos_sim.shape[1]].sum(dim=-1)
                batch_scores.append(pos_score.unsqueeze(-1))
            batch_scores = torch.cat(batch_scores, dim=-1)

            batch_scores = batch_scores.mean(dim=-1)
            all_scores.extend(batch_scores.cpu().numpy())

    all_scores = np.array(all_scores)
    print(f"{all_scores.shape= }")
    return all_scores


def neglabel(model_path="pretrain"):
    recall = 0.95
    batch_size = 256
    shuffle = True
    arch = "ViT-B/16"

    method = "NegLabel"

    with open(f"{model_path}/{method}.csv", "w") as fw:
        fw.write(f"{arch} {model_path.split('/')}")

        clip_model, preprocess = clip.load(arch, device, jit=False)
        if not model_path == "pretrain":
            if os.path.exists(model_path):
                checkpoint = torch.load(f"{model_path}/checkpoint.pth", weights_only=False)
                clip_model.load_state_dict(checkpoint["model_state_dict"])

                print("Loaded the best model for final evaluation.")
        clip_model.eval()

        dataset_val = ImageTextDataset(
            f"data/imagenet/",
            f"datalists/imagenet_val_20.txt",
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

        prompt_idxs = [85]
        emb_batchsize = 500
        pencentile = 0.95
        neg_topk = 0.15
        ngroup = 100

        for prompt_idx_pos in prompt_idxs:
            prompt_idx_neg = prompt_idx_pos

            train_dataset = "imagenet"
            class_name = CLASS_NAME[train_dataset]
            prompts = [prompt_templates[prompt_idx_pos].format(c) for c in class_name]
            text_inputs = torch.cat([clip.tokenize(f"{c}") for c in prompts]).to(device)

            with torch.no_grad():
                text_features_pos = clip_model.encode_text(text_inputs).to(torch.float32)
                text_features_pos /= text_features_pos.norm(dim=-1, keepdim=True)

            words_noun = []
            words_adj = []
            prompt_templete = dict(
                adj="This is a {} photo",
                noun=prompt_templates[prompt_idx_neg],
            )

            wordnet_database = f"txtfiles"
            txtfiles = os.listdir(wordnet_database)

            dedup = dict()
            for file in txtfiles:
                filetype = file.split(".")[0]
                if filetype not in prompt_templete:
                    continue
                with open(os.path.join(wordnet_database, file), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip() in dedup:
                            continue
                        dedup[line.strip()] = None
                        if filetype == "noun":
                            words_noun.append(prompt_templete[filetype].format(line.strip()))
                        elif filetype == "adj":
                            words_adj.append(prompt_templete[filetype].format(line.strip()))
                        else:
                            raise TypeError

            text_inputs_neg_noun = torch.cat([clip.tokenize(f"{c}") for c in words_noun]).to(device)
            text_inputs_neg_adj = torch.cat([clip.tokenize(f"{c}") for c in words_adj]).to(device)
            text_inputs_neg = torch.cat([text_inputs_neg_noun, text_inputs_neg_adj], dim=0)
            noun_length = len(text_inputs_neg_noun)
            adj_length = len(text_inputs_neg_adj)

            with torch.no_grad():
                text_features_neg = []
                for i in range(0, len(text_inputs_neg), emb_batchsize):
                    x = clip_model.encode_text(text_inputs_neg[i : i + emb_batchsize]).to(torch.float32)
                    text_features_neg.append(x)
                text_features_neg = torch.cat(text_features_neg, dim=0)
                text_features_neg /= text_features_neg.norm(dim=-1, keepdim=True)

                neg_sim = []
                for i in range(0, noun_length + adj_length, emb_batchsize):
                    tmp = text_features_neg[i : i + emb_batchsize] @ text_features_pos.T
                    tmp = tmp.to(torch.float32)
                    # 计算输入张量沿指定维度的分位数（quantiles）
                    sim = torch.quantile(tmp, q=pencentile, dim=-1)
                    neg_sim.append(sim)

                neg_sim = torch.cat(neg_sim, dim=0)
                neg_sim_noun = neg_sim[:noun_length]
                neg_sim_adj = neg_sim[noun_length:]
                text_features_neg_noun = text_features_neg[:noun_length]
                text_features_neg_adj = text_features_neg[noun_length:]

                ind_noun = torch.argsort(neg_sim_noun)
                ind_adj = torch.argsort(neg_sim_adj)
                text_features_neg = torch.cat(
                    [
                        text_features_neg_noun[ind_noun[0 : int(len(ind_noun) * neg_topk)]],
                        text_features_neg_adj[ind_adj[0 : int(len(ind_adj) * neg_topk)]],
                    ],
                    dim=0,
                )

            fw.write(
                f"{method}, prompt_idx_pos: {prompt_idx_pos}, prompt_idx_neg: {prompt_idx_neg}, penc: {pencentile}, neg_topk: {neg_topk}, ngroup: {ngroup}\n"
            )

            scores_id = detection(
                text_features_pos,
                text_features_neg,
                dataloader_val,
                clip_model,
                "id_val",
                ngroup=ngroup,
                random_permute=True,
            )

            fpr_all = 0
            auroc_all = 0
            for ood_name, dataloader_ood in zip(ood_names, dataloader_oods):
                scores_ood = detection(
                    text_features_pos,
                    text_features_neg,
                    dataloader_ood,
                    clip_model,
                    ood_name,
                    ngroup=ngroup,
                    random_permute=True,
                )

                auc_ood = auc(scores_id, scores_ood)[0]
                fpr_ood, _ = fpr_recall(scores_id, scores_ood, recall)
                fpr_all += fpr_ood
                auroc_all += auc_ood

                fw.write(f"auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n")
            fw.write(f"mean auroc {auroc_all/len(ood_names):.2%}, mean fpr {fpr_all/len(ood_names):.2%}\n\n")


if __name__ == "__main__":
    neglabel()
