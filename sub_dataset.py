import random
from collections import defaultdict


def select_k_samples_per_class(input_file, output_file, k, seed=45):
    """
    从输入文件中为每个类别随机选择k个样本，并写入输出文件

    参数:
    input_file (str): 输入txt文件路径
    output_file (str): 输出txt文件路径
    k (int): 每个类别选择的样本数
    seed (int): 随机数生成器种子，确保结果可重现
    """
    # 设置随机数种子
    random.seed(seed)

    # 按类别分组样本
    class_samples = defaultdict(list)

    # 读取文件并按类别分组
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)  # 从右侧分割，确保只分割最后一个空格
            if len(parts) != 2:
                print(f"警告: 格式不正确的行被忽略: {line}")
                continue
            path, label = parts
            class_samples[label].append(path)

    # 为每个类别随机选择k个样本
    selected_samples = []
    for label, samples in class_samples.items():
        if len(samples) < k:
            print(f"警告: 类别 {label} 只有 {len(samples)} 个样本，不足 {k} 个")
            selected = samples  # 样本不足时选择全部
        else:
            selected = random.sample(samples, k)
        # 添加到结果列表
        for path in selected:
            selected_samples.append(f"{path} {label}")

    # 随机打乱所有选中的样本
    random.shuffle(selected_samples)

    # 写入输出文件
    with open(output_file, "w") as f:
        f.write("\n".join(selected_samples) + "\n")

    print(f"已成功为每个类别选择 {k} 个样本，并保存到 {output_file}")
    print(f"总样本数: {len(selected_samples)}")
    print(f"类别数: {len(class_samples)}")


# 使用示例
if __name__ == "__main__":

    k = 16  # 每个类别选择的样本数
    input_file = "datalists/imagenet_r_train.txt"  # 替换为你的输入文件路径
    output_file = f"datalists/imagenet_r_{k}shot_3.txt"  # 替换为你的输出文件路径

    select_k_samples_per_class(input_file, output_file, k, 2)
