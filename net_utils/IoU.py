import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from collections import Counter

def calculate_iou(pred_mask, label_mask):
    intersection = np.logical_and(pred_mask, label_mask)
    union = np.logical_or(pred_mask, label_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou
def calculate_accuracy(pred_mask, label_mask):
    correct_pixels = np.sum(pred_mask == label_mask)
    total_pixels = pred_mask.size
    accuracy = correct_pixels / total_pixels
    return accuracy
def integral_label(predictions):
    """
    整合聚类群组的整体语义标签
    :param predictions: 按行排列的输出标签形状为N*1的列表
    """
    label_mapping = {'pedestrian': 0, 'vehicle': 1, 'ghost': 2}
    flattened_data = [item for sublist in predictions for item in sublist]
    # Use Counter to count occurrences
    counter = Counter(flattened_data)
    # Find the most common element
    most_common_label, most_common_count = counter.most_common(1)[0]
    for key,label in label_mapping.items():
        if label == most_common_label:
            return key
def integral_compute_metrics(predictions, labels, seg_classes):
    """
    """
    class_iou = {}
    class_accuracy = {}
    confusion_matrix = np.zeros((len(seg_classes['Scenes']), len(seg_classes['Scenes'])), dtype=np.int32)
    # confusion_matrix_percentage = np.zeros((len(seg_classes['Scenes']), len(seg_classes['Scenes'])))
    pred_mask_temp = np.empty((0, predictions.shape[1]))
    label_mask_temp = np.empty((0, labels.shape[1]))
    for class_id in seg_classes['Scenes']:
        pred_mask = (predictions == class_id)
        label_mask = (labels == class_id)

        iou = calculate_iou(pred_mask, label_mask)
        accuracy = calculate_accuracy(pred_mask, label_mask)

        class_iou[class_id] = iou
        class_accuracy[class_id] = accuracy

        # Update confusion matrix
        for i in range(len(seg_classes['Scenes'])):
            confusion_matrix[i][class_id] = np.sum(np.logical_and(predictions == class_id, labels == i))
            # 模糊矩阵每个元素为 满足两个条件的布尔类型矩阵中的True的个数求和.
        pred_mask_temp = np.concatenate((pred_mask_temp, pred_mask), axis=0)
        label_mask_temp = np.concatenate((label_mask_temp, label_mask), axis=0)
    total_iou = calculate_iou(pred_mask_temp, label_mask_temp)
    total_accuracy = calculate_accuracy(pred_mask_temp, label_mask_temp)
    print("Class IoU:")
    for class_id, iou in class_iou.items():
        print(f"Class {class_id}: {iou:.4f}")

    print("Class Accuracy:")
    for class_id, accuracy in class_accuracy.items():
        print(f"Class {class_id}: {accuracy:.4f}")

    print("Total IoU:", total_iou)
    print("Total Accuracy:", total_accuracy)

    print("Confusion Matrix:")
    print(confusion_matrix)

    print("Confusion Matrix (percentages):")
    confusion_matrix_percentage = confusion_matrix.astype(np.float32)
    # confusion_matrix_percentage=confusion_matrix
    for i in range(len(seg_classes['Scenes'])):
        confusion_matrix_percentage[i] /= np.sum(confusion_matrix_percentage[i])
    confusion_matrix_percentage *= 100
    # keep the decimal place to one place
    confusion_matrix_percentage = np.round(confusion_matrix_percentage, 1)
    print(confusion_matrix_percentage)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_percentage, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=['pedestrian', 'vehicle', 'ghost'],
                yticklabels=['pedestrian', 'vehicle', 'ghost']) ##, fmt=".1f%"

    # # 手动添加百分比符号
    # for i in range(confusion_matrix.shape[0]):
    #     for j in range(confusion_matrix.shape[1]):
    #         plt.text(j + 0.5, i + 0.5, f"{confusion_matrix[i, j]:.1f}%", ha='center', va='center')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    return total_iou, total_accuracy,confusion_matrix,confusion_matrix_percentage

def integral_evaluate(scene_name):
    label_mapping = {'pedestrian': 0, 'vehicle': 1, 'ghost': 2}
    seg_classes={'Scenes':[0,1,2]}
    label_path=f'DataSets/self/cluster_{scene_name}.pkl'
    prediction_path=f'InputData/SCnew/LabelCluster_{scene_name}.pkl'
    with open(label_path, 'rb') as f:
        labels = pickle.load(f)
    with open(prediction_path, 'rb') as f:
        predictions = pickle.load(f)
    truth_labels=[]
    truth_predictions=[]
    for labels_frame,predictions_frame in zip(labels,predictions):
        for label,prediction in zip(labels_frame,predictions_frame):
            truth_labels.append([label_mapping[label[1]]])
            truth_predictions.append([label_mapping[prediction[1]]])
    total_iou, total_accuracy,confusion_matrix,confusion_matrix_percentage=integral_compute_metrics(np.array(truth_labels),np.array(truth_predictions),seg_classes)
    return total_iou,total_accuracy,confusion_matrix,confusion_matrix_percentage
def compute_metrics(predictions, labels, seg_classes):
    class_iou = {}
    class_accuracy = {}
    confusion_matrix = np.zeros((len(seg_classes['Scenes']), len(seg_classes['Scenes'])), dtype=np.int32)
    pred_mask_temp = np.empty((0, predictions.shape[1]))
    label_mask_temp = np.empty((0, labels.shape[1]))
    for class_id in seg_classes['Scenes']:
        pred_mask = (predictions == class_id)
        label_mask = (labels == class_id)

        iou = calculate_iou(pred_mask, label_mask)
        accuracy = calculate_accuracy(pred_mask, label_mask)

        class_iou[class_id] = iou
        class_accuracy[class_id] = accuracy

        # Update confusion matrix
        for i in range(len(seg_classes['Scenes'])):
            confusion_matrix[i][class_id] = np.sum(np.logical_and(predictions == class_id, labels == i))
            #模糊矩阵每个元素为 满足两个条件的布尔类型矩阵中的True的个数求和.
        pred_mask_temp = np.concatenate((pred_mask_temp, pred_mask), axis=0)
        label_mask_temp = np.concatenate((label_mask_temp, label_mask), axis=0)
        if class_id < 2:
            pred_mask_temp_without_static = pred_mask_temp
            label_mask_temp_without_static = label_mask_temp

    total_iou = calculate_iou(pred_mask_temp, label_mask_temp)
    total_accuracy = calculate_accuracy(pred_mask_temp, label_mask_temp)
    total_iou_without_static = calculate_iou(pred_mask_temp_without_static, label_mask_temp_without_static)
    total_accuracy_without_static = calculate_accuracy(pred_mask_temp_without_static, label_mask_temp_without_static)

    print("Class IoU:")
    for class_id, iou in class_iou.items():
        print(f"Class {class_id}: {iou:.4f}")

    print("Class Accuracy:")
    for class_id, accuracy in class_accuracy.items():
        print(f"Class {class_id}: {accuracy:.4f}")

    print("Total IoU:", total_iou)
    print("Total Accuracy:", total_accuracy)

    print("Total IoU without static:", total_iou_without_static)
    print("Total Accuracy without static:", total_accuracy_without_static)

    print("Confusion Matrix:")
    print(confusion_matrix)

    print("Confusion Matrix (percentages):")
    confusion_matrix = confusion_matrix.astype(np.float32)

    for i in range(len(seg_classes['Scenes'])):
        confusion_matrix[i] /= np.sum(confusion_matrix[i])
    confusion_matrix *= 100
    # keep the decimal place to one place
    confusion_matrix = np.round(confusion_matrix, 1)
    print(confusion_matrix)

    return total_iou, total_accuracy,confusion_matrix
def plot_confusion_matrix(confusion_matrix, class_names,savepath,if_save):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt=".1f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    if if_save:
        directory, filename = os.path.split(savepath)
        filename_without_ext = os.path.splitext(filename)[0]
        # 保存图片到指定路径，同名文件加png扩展名
        new_file_path = os.path.join(directory, f"{filename_without_ext}.png")
        plt.savefig(new_file_path)

if __name__=="__main__":
    seg_classes = {'Scenes': [0, 1, 2]}
    predictions = np.random.randint(0, 3, size=(4,3, 100))
    labels = np.random.randint(0, 3, size=(4,3, 100))

    iou, acc, conf_matrix = compute_metrics(np.concatenate(predictions, axis=0), np.concatenate(labels, axis=0),
                                            seg_classes)
    plot_confusion_matrix(conf_matrix, seg_classes['Scenes'])