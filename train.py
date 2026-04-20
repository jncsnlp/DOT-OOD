import torch
import os
import clip

from class_names import CLASS_NAME, prompt_templates
from torch.utils.data import DataLoader
from build_dataset import ImageTextDataset
import time
import ot
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from sub_dataset import select_k_samples_per_class
from ood_dete_csp import csp


def optimal_transport(image, text, epsilon):
    S = 1 - torch.matmul(image.view(-1, 1), text.view(1, -1))

    a = (torch.ones(S.shape[0]) / S.shape[0]).to(device)
    b = (torch.ones(S.shape[0]) / S.shape[0]).to(device)

    T = ot.sinkhorn(a, b, S, epsilon, numItermax=1000, stopThr=1e-8)

    image_aligned = torch.matmul(T.T, image)
    image_aligned = image_aligned / torch.norm(image_aligned, p=2, dim=-1, keepdim=True)

    return image_aligned


def load_dataset(shot, batch_size, round):
    # 构造数据集和数据加载器
    output_file = f"datalists/imagenet2012_{shot}shot_{round}.txt"
    select_k_samples_per_class("datalists/imagenet2012_train.txt", output_file, shot, round)

    logger.info("Loading Datasets")
    dataset_train = ImageTextDataset(
        f"data/imagenet",
        output_file,
        preprocess,
    )
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = ImageTextDataset(
        f"data/imagenet/",
        f"datalists/imagenet2012_val.txt",
        preprocess,
    )
    dataloader_val = DataLoader(dataset_val, batch_size=4 * batch_size, shuffle=True)

    logger.info(f"{len(dataloader_train)= }")
    logger.info(f"{len(dataloader_val)= }")

    return dataloader_train, dataloader_val


def get_logger(path):
    # 移除之前的日志处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_file = f"{path}/training_log.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )
    logger = logging.getLogger()

    return logger


def load_clip_model():
    # 默认冻结所有参数
    for param in clip_model.parameters():
        param.requires_grad = False

    for name, param in clip_model.named_parameters():
        if (
            "visual.transformer.resblocks.10" in name
            or "visual.transformer.resblocks.11" in name
            or "visual.proj" in name
            or "transformer.resblocks.10" in name
            or "transformer.resblocks.11" in name
            or "text_projection" in name
        ):
            param.requires_grad = True
            logger.info(name)
        else:
            param.requires_grad = False

    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in clip_model.parameters())
    logger.info(f"Trainable parameters: {trainable_params}/{total_params} ({trainable_params/total_params:.2%})")

    return clip_model


def train(dataloader_train, dataloader_val, epsilon, alpha, beta):
    logger.info(
        f"Training CLIP-B/16 Model, Hyperparameter are epsilon ({epsilon}), alpha ({alpha}), beta ({beta}) and temperature ({t}). The Learning Rate is {learning_rate}, and the number of epoch is {num_epochs}.\n"
    )

    clip_model = load_clip_model()

    optimizer = torch.optim.SGD(clip_model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_accuracy = 0.0

    logger.info(f"{method}: {epsilon}_{alpha}_{beta}_{t}", 50 * "-")
    for epoch in range(num_epochs):
        clip_model.train()
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch + 1}: Current learning rate = {current_lr}")

        all_correct = 0
        loss_all = 0
        all_time = 0
        loss_ot_all = 0
        loss_neg_all = 0
        clip_model.train()
        for i, (imgs, labels) in enumerate(dataloader_train):
            time_start = time.time()
            # 前向传播
            image_input = imgs.to(device)
            labels = labels.to(device)
            prompts = [prompt_templates[prompt_idx_pos].format(c) for c in class_name]
            text_inputs = torch.cat([clip.tokenize(f"{c}") for c in prompts]).to(device)

            image_features, local_image_features = clip_model.encode_image(image_input)
            image_features = image_features.to(torch.float32)
            local_image_features = local_image_features.to(torch.float32)

            text_features = clip_model.encode_text(text_inputs).to(torch.float32)

            # 归一化特征
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logits = (image_features @ text_features.T) / t

            loss_ot = 0
            loss_neg = 0
            for image_feature, label in zip(image_features, labels):
                align_image_feature = optimal_transport(image_feature, text_features[label], epsilon)
                loss_ot += (1 - align_image_feature @ text_features[label]) / len(labels)

                image_neg = image_features[labels != label]
                loss_neg += torch.sum(abs(align_image_feature @ image_neg.T)) / len(image_neg)
            #
            # 计算损失
            loss = criterion(logits, labels) + alpha * loss_ot + beta * loss_neg

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all += loss.item()
            loss_ot_all += loss_ot.item()
            loss_neg_all += loss_neg.item()

            _, preds = torch.max(logits, dim=1)
            correct = torch.sum(preds == labels).item()
            all_correct += correct

            time_end = time.time()
            batch_time = time_end - time_start
            all_time += batch_time

            print(
                f"\rTraining ({method}) - Epoch {epoch + 1} / {num_epochs}, Batch: {i} / {len(dataloader_train)}, Loss: {loss.item():.3f}, Loss_ot: {loss_ot.item():.3f}, Loss_neg: {loss_neg.item():.3f}, Correct: {correct}/{len(imgs)}, Time:{batch_time:.3f}",
                end="",
                flush=True,
            )

        # 计算该 epoch 的平均损失
        epoch_loss = loss_all / len(dataloader_train)
        epoch_loss_ot = loss_ot_all / len(dataloader_train)
        epoch_loss_neg = loss_neg_all / len(dataloader_train)

        logger.info(
            f"Trained ({method}) - Epoch {epoch + 1}: Loss: {epoch_loss:.4f}, Loss_ot: {epoch_loss_ot:.4f}, Loss_neg: {epoch_loss_neg:.4f}, Correct: {all_correct}, Time: {all_time:.4f}"
        )

        # 验证集评估
        val_accuracy = evaluate(clip_model, dataloader_val, class_name, prompt_idx_pos, epoch, num_epochs, method)
        logger.info(f"Validation: epoch= {(epoch + 1)}, Acc= {val_accuracy:.5f}\n")

        if epoch == num_epochs - 1:
            torch.save({"model_state_dict": clip_model.state_dict()}, f"{save_model_path}/checkpoint.pth")

        # 更新学习率
        scheduler.step()


def evaluate(model, dataloader, class_name, prompt_idx_pos, epoch, num_epochs, method):
    model.eval()
    val_correct = 0
    val_total = 0
    prompts = [prompt_templates[prompt_idx_pos].format(c) for c in class_name]
    text_inputs = torch.cat([clip.tokenize(f"{c}") for c in prompts]).to(device)
    time_val = 0
    with torch.no_grad():
        text_features = model.encode_text(text_inputs).to(torch.float32)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        for i, (imgs, labels) in enumerate(dataloader):
            time_val_s = time.time()
            image_input = imgs.to(device)
            labels = labels.to(device)
            image_features, _ = model.encode_image(image_input)
            image_features = image_features.to(torch.float32)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            _, preds = torch.max(logits, dim=1)
            correct = torch.sum(preds == labels).item()
            val_correct += correct
            val_total += len(imgs)
            time_val_e = time.time()
            epoch_time = time_val_e - time_val_s
            time_val += epoch_time

            print(
                f"\rid-val ({method}) - Epoch {(epoch + 1)} / {num_epochs}: {i} / {len(dataloader)}, correct = {val_correct}, acc = {(val_correct / val_total):.4f}, time = {epoch_time:.4f}",
                end="",
                flush=True,
            )
    val_accuracy = val_correct / val_total

    return val_accuracy


device = "cuda" if torch.cuda.is_available() else "cpu"
arch = "ViT-B/16"
prompt_idx_pos = 39
t = 0.01
perfroer = [0.3, 0.5, 0.5]
num_epochs = 15
learning_rate = 2e-3
criterion = torch.nn.CrossEntropyLoss()
train_dataset = "imagenet"
class_name = CLASS_NAME[train_dataset]
shot = 16
batch_size = 128
epsilon, alpha, beta = perfroer[0], perfroer[1], perfroer[2]
method = "train_cls_ot_otloss_negloss"

save_path = f"trained_model/{arch}/{shot}shot/{epsilon}_{alpha}_{beta}/{batch_size}/{method}"
logger = get_logger(save_path)

logger.info(f"Loading CLIP model ({arch}) on {device}")
try:
    clip_model, preprocess = clip.load(arch, device, jit=False)
    if clip_model is None:
        raise ValueError("Failed to load CLIP model")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise


if __name__ == "__main__":
    logger.info(f"{prompt_templates[prompt_idx_pos]= }")
    for i in range(0, 3):
        save_model_path = f"{save_path}_{i}"
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        dataloader_train, dataloader_val = load_dataset(shot, batch_size, i)
        logger.info("Start the {i}th training", f"{'*'*100}")
        train(dataloader_train, dataloader_val, epsilon, alpha, beta)
        csp(save_model_path)
