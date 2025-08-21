import sys, re


def correctAnswerStart(fileName):
    """修正数据集中部分"answer_start"的值是字符串而非整型的问题

    Args:
        fileName (_type_): 要修正的文件名
    """
    # 读取需要修改的文件
    with open(fileName, "r", encoding="utf-8") as f:
        json_str = f.read()

    # 寻找并修改
    pattern = r'"answer_start": "(\d+)"'
    replace_str = re.sub(pattern, r'"answer_start": \1', json_str)

    # 写入修改后的文件
    with open(fileName, "w", encoding="utf-8") as f:
        f.write(replace_str)


if __name__ == "__main__":
    correctAnswerStart(sys.argv[1])
