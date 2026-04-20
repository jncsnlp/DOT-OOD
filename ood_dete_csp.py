import torch
import os
import numpy as np
import random
import clip
from class_names import (
    CLASS_NAME,
    preset_noun_prompt_templates,
    csp_templates,
    preset_adj_prompt_templates,
)
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from build_dataset import ImageTextDataset
from measure import auc, fpr_recall
import itertools

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
    return all_scores


def csp(model_path="pretrain"):
    recall = 0.95
    batch_size = 256
    shuffle = True
    arch = "ViT-B/16"

    method = "CSP"

    with open(f"{model_path}/{method}.csv", "w") as fw:
        fw.write(f"{arch} {model_path.split('/')}\n")

        clip_model, preprocess = clip.load(arch, device, jit=False)
        if not model_path == "pretrain":
            if os.path.exists(model_path):
                checkpoint = torch.load(f"{model_path}/checkpoint.pth", weights_only=False)
                clip_model.load_state_dict(checkpoint["model_state_dict"])
                fw.write(f"Loaded the retrained for final evaluation.")
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

        emb_batchsize = 500
        pencentile = 0.95
        neg_topk = 0.15
        ngroup = 100
        seed = 0

        train_dataset = "imagenet"
        class_name = CLASS_NAME[train_dataset]

        id_prompts = [pair[0].format(pair[1]) for pair in list(itertools.product(preset_noun_prompt_templates, class_name))]

        # 【分批处理】：分批次 tokenize + 编码
        text_features_pos_list = []
        for i in range(0, len(id_prompts), emb_batchsize):
            # 1. 分批取 prompt
            batch_prompts = id_prompts[i : i + emb_batchsize]
            # 2. 分批 tokenize
            batch_tokens = [clip.tokenize(c) for c in batch_prompts]
            batch_tokens = torch.cat(batch_tokens).to(device)
            # 3. 分批编码
            with torch.no_grad():
                batch_feat = clip_model.encode_text(batch_tokens).to(torch.float32)
                text_features_pos_list.append(batch_feat)
            # 【可选】主动释放当前批次显存
            del batch_tokens, batch_feat
            torch.cuda.empty_cache()

        # 4. 拼接所有批次的特征
        text_features_pos = torch.cat(text_features_pos_list, dim=0)
        feat_dim = text_features_pos.shape[-1]
        # 5. 恢复原逻辑的维度处理
        text_features_pos = text_features_pos.view(-1, len(class_name), feat_dim).mean(dim=0)
        text_features_pos /= text_features_pos.norm(dim=-1, keepdim=True)

        words_noun = []
        words_adj = []
        prompt_templete = dict(
            adj=csp_templates,
            noun=preset_noun_prompt_templates,
        )

        txt_exclude = "noun.person.txt,noun.quantity.txt,noun.group.txt,adj.pert.txt"
        wordnet_database = f"txtfiles"
        txtfiles = os.listdir(wordnet_database)

        file_names = txt_exclude.split(",")
        for file in file_names:
            txtfiles.remove(file)

        dedup = dict()
        random.seed(seed)
        noun_length = 0
        adj_length = 0

        for file in txtfiles:
            filetype = file.split(".")[0]
            if filetype not in prompt_templete:
                continue
            with open(os.path.join(wordnet_database, file), "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.replace("_", " ")
                    if line.strip() in dedup:
                        continue
                    dedup[line.strip()] = None
                    if filetype == "noun":
                        noun_length += 1
                        for template in prompt_templete[filetype]:
                            words_noun.append(template.format(line.strip()))
                    elif filetype == "adj":
                        adj_length += 1
                        candidate = random.choice(prompt_templete[filetype]).format(line.strip())
                        for template in preset_adj_prompt_templates:
                            words_adj.append(template.format(candidate))
                    else:
                        raise TypeError

        # 【分批优化】：负样本文本编码改为分批处理
        text_features_neg = []
        # 先处理 noun 负样本
        for i in range(0, len(words_noun), emb_batchsize):
            batch_noun = words_noun[i : i + emb_batchsize]
            text_inputs_batch = torch.cat([clip.tokenize(c) for c in batch_noun]).to(device)
            with torch.no_grad():
                feat_batch_noun = clip_model.encode_text(text_inputs_batch).to(torch.float32)

                text_features_neg.append(feat_batch_noun)
            del text_inputs_batch, feat_batch_noun
            torch.cuda.empty_cache()

        # 再处理 adj 负样本
        for i in range(0, len(words_adj), emb_batchsize):
            batch_adj = words_adj[i : i + emb_batchsize]
            text_inputs_batch = torch.cat([clip.tokenize(c) for c in batch_adj]).to(device)
            with torch.no_grad():
                feat_batch_adj = clip_model.encode_text(text_inputs_batch).to(torch.float32)
                # 按模板维度平均
                text_features_neg.append(feat_batch_adj)
            del text_inputs_batch, feat_batch_adj
            torch.cuda.empty_cache()

        # 拼接所有负样本特征（此时已是分批编码后的结果）
        text_features_neg = torch.cat(text_features_neg, dim=0)
        feat_dim = text_features_neg.shape[-1]
        text_features_neg = text_features_neg.view(-1, len(words_noun) + len(words_adj), feat_dim).mean(dim=0)
        text_features_neg /= text_features_neg.norm(dim=-1, keepdim=True)

        # 【分批优化】：相似度计算也分批，避免大矩阵一次性计算
        neg_sim = []
        total_neg = len(text_features_neg)
        for i in range(0, total_neg, emb_batchsize):
            batch_neg_feat = text_features_neg[i : i + emb_batchsize]
            # 计算当前批次负样本与正样本的相似度
            tmp_sim = batch_neg_feat @ text_features_pos.T
            tmp_sim = tmp_sim.to(torch.float32)

            # 分位数计算
            sim_quantile = torch.quantile(tmp_sim, q=pencentile, dim=-1)
            # 最大值筛选
            max_sim, _ = torch.max(tmp_sim, dim=1)
            sim_quantile[max_sim > 0.95] = 1.0

            neg_sim.append(sim_quantile)

        neg_sim = torch.cat(neg_sim, dim=0)
        # 拆分 noun/adj 相似度
        neg_sim_noun = neg_sim[:noun_length]
        neg_sim_adj = neg_sim[noun_length:]
        text_features_neg_noun = text_features_neg[:noun_length]
        text_features_neg_adj = text_features_neg[noun_length:]

        # 筛选 TopK
        ind_noun = torch.argsort(neg_sim_noun)
        ind_adj = torch.argsort(neg_sim_adj)

        text_features_neg_noun_selected = text_features_neg_noun[ind_noun[0 : int(len(ind_noun) * neg_topk)]]
        text_features_neg_adj_selected = text_features_neg_adj[ind_adj[0 : int(len(ind_adj) * neg_topk)]]
        text_features_neg = torch.cat([text_features_neg_noun_selected, text_features_neg_adj_selected], dim=0)

        fw.write(f"{method}, penc: {pencentile}, neg_topk: {neg_topk}, ngroup: {ngroup}\n")

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
            # break
        fw.write(f"mean auroc {auroc_all/len(ood_names):.2%}, mean fpr {fpr_all/len(ood_names):.2%}\n\n")


if __name__ == "__main__":
    csp()
