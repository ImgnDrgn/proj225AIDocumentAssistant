import json

filepath = "../数据集/数据集合并/DeepinWiki.json"
endfile = "./DeepinWiki.txt"

with open(endfile, "w", encoding="utf-8") as fe:
    with open(filepath, "r", encoding="utf-8") as fw:
        wiki = json.load(fw)
        data_list = wiki["data"]
        for data in data_list:
            fe.write("标题：" + data["title"] + "\n")
            paragraphs_list = data["paragraphs"]
            for paragraphs in paragraphs_list:
                context = paragraphs["context"]
                fe.write(context + "\n\n")
