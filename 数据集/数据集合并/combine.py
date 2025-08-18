import os, json, sys


def combine_all_json(doc_dir: str, combined_file: str):
    """将所有json文件合并为一个，从而用于模型训练

    Args:
        doc_dir (str): 待合并的文档所在的目录，注意：所有文档须放在同一目录下，该目录下不能有子目录等其他内容
        combined_file (str): 合并后的json文件路径，注意：以.json结尾
    """
    with open(
        combined_file, "w", encoding="utf-8"
    ) as combined:  # 打开合并后的文件，用于写入

        data = (
            []
        )  # 用于存放"data"键下所有内容，每一篇文档的内容都是一个字典，包含"title"和"paragraphs"两个键。这个字典作为data列表的元素
        title_dict = (
            dict()
        )  # 用于存放所有出现过的标题及其出现次数，从而处理某些文档中"title"重复的问题

        for filename in os.listdir(
            doc_dir
        ):  # 列出目录下所有文件（未检测是否是json、是否是有效目录，使用时需要将所有json文档放到一个有效目录下）

            file_path = os.path.join(doc_dir, filename)  # 组成文件完整路径

            # 只读打开要处理的json文件
            with open(file_path, "r", encoding="utf-8") as file:
                # 可能存在wiki文档缺失，用空文件占位的情况
                try:
                    # 对于非空文件
                    json_content = json.load(file)  # 将json文件内容转换成字典
                    try:
                        # 绝大部分应该是有data键的，但是部分可能一上来就直接"title"和"paragraphs"了
                        # 对于有"data"键的，其值是只有一个元素的列表，直接和data列表合并

                        # 合并前需要查看这个文档的"title"是否出现过了
                        title = json_content["data"][0][
                            "title"
                        ]  # 读取json中"title"的内容
                        if title in title_dict:
                            # 如果已经出现过，计数+1，并且修改"title"
                            title_dict[title] = title_dict[title] + 1
                            title = f"{title}-{title_dict[title]}"  # 名字改为"原title-出现次数"
                            json_content["data"][0][
                                "title"
                            ] = title  # 修改json_content中的"title"
                        else:
                            # 对于首次出现的标题，将其加入title_dict，并计数为1
                            title_dict[title] = 1

                        # 再合并本篇文档
                        data = data + json_content["data"]
                    except KeyError:
                        # 对于没有"data"键的，上来是"title和"paragraphs"，直接将json_content作为元素添加到data列表中

                        # 同样合并前先判断是否出现过
                        title = json_content["title"]
                        if title in title_dict:
                            title_dict[title] = title_dict[title] + 1
                            title = f"{title}-{title_dict[title]}"
                            json_content["title"] = title
                        else:
                            title_dict[title] = 1

                        data.append(json_content)

                except json.JSONDecodeError:
                    pass  # 对于空文件，不进行操作，自行关闭后处理下一个即可

        # 所有文件提取完毕后，将其分装成json格式的字典
        json.dump(dict(data=data), combined)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("参数数量错误！")
    else:
        combine_all_json(sys.argv[1], sys.argv[2])
        print("合并成功！")
