import math
from collections import Counter


def x_dot(x, y, r_dot, y_dot=1):
    """
    Calculate the value of x based on the inverse functions of the H function.

    Parameters:
    r_dot (float): radial velocity.
    x (float): The x-coordinate.
    y (float): The y-coordinate.
    y_dot (float): The velocity y, set to 1 based on experience.

    Returns:
    float: The value of x_dot.
    """
    return (r_dot * math.sqrt(x ** 2 + y ** 2) - y * y_dot) / x


def has_different_frequencies(labels):
    counter = Counter(labels)
    frequency_values = list(counter.values())
    if len(frequency_values) == 1 and frequency_values[0] > 1:
        # 如果一个轨迹标签都为0，则需要返回真，对其进行判断
        return True
    else:
        # 如果所有元素的出现次数都相同，则返回 False，否则返回 True
        return not all(frequency == frequency_values[0] for frequency in frequency_values)


def get_label_id(labels):
    counter = Counter(labels)
    most_common = counter.most_common(1)
    label_id = most_common[0][0]
    return label_id


if __name__ == '__main__':
    labels = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print(has_different_frequencies(labels))
    print(get_label_id(labels))
