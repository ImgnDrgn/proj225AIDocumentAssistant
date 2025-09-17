# 知识库构建与使用

## 知识库生成文本

使用[contextExtract.py](./contextExtract.py)将最终版的[DeepinWiki.json](../数据集/数据集合并/DeepinWiki.json)之中的标题和正文提取出来，合并在一个文本[DeepinWiki.txt](./DeepinWiki.txt)中，用于知识库的生成

## 知识库生成工具

预计使用 ollama+AnythingLLM 实现在本地构建知识库并用于本地模型使用

## 最终方案
对上面的方法进行了各种尝试，但还是失败了 ~~QAQ~~

最终决定自己手搓，运行`train.py`即可 *需要修改本地模型的位置*
