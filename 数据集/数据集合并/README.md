# 本目录存放合并数据集相关内容

## 脚本说明
[combine.py]()用于将生成的所有json文件合并为一个，从而满足模型训练需求

### 使用方法
`python combine.py [doc_dir] [combined_file]`
### 参数说明
`doc_dir`：待合并的文档所在的目录 ***注意：所有文档须放在同一目录下，该目录下不能有子目录等其他内容***

`combined_file`：合并后的json文件路径 ***注意：以.json结尾***

### 示例
```python combine.py "./docs" "./test.json"```
合并和文档内容如下（已折叠部分内容）
<img width="1400" height="731" alt="image" src="https://github.com/user-attachments/assets/1403ec84-6f33-4ad7-9dc7-175d3c38c87b" />

## 合并后文档说明
*未完成*
