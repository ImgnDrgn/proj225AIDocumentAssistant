# 说明
## 文档范围
选取的文档范围是[wiki](https://wiki.deepin.org/zh/home)目录中**01_软件wiki**到**05_HOW-TO** ***（不包含03_技术规范）*** 以及**待分类**中的全部文档，文档从[wiki的github仓库](https://github.com/linuxdeepin/wiki.deepin.org.git)获得

## 文档转换流程
1. 下载对应文档 *下载的文档以**2025年7月27日**版本为准*
2. 分别交给两个ai，进行转换
3. 将两个数据集进行合并
   > 合并要求
   > 1. 一篇文档仅对应一个title
   > 2. 问题id格式如下：`1-1-1`
   >    > 其中第一个数字是文档所在大文件夹编号，**01_软件wiki**为1，以此类推，**待分类**是5
   >    > 第二个数字是文档在该大文件夹中的顺序（在[wiki官网](https://wiki.deepin.org/zh/home)中的顺序，从1开始），如文档[CherryTree](https://wiki.deepin.org/zh/01_%E8%BD%AF%E4%BB%B6wiki/00_GUI%E8%BD%AF%E4%BB%B6/03_%E7%AC%AC%E4%B8%89%E6%96%B9%E5%BC%80%E5%8F%91%E7%9A%84%E8%BD%AF%E4%BB%B6/00_%E5%8A%9E%E5%85%AC%E5%95%86%E4%B8%9A%E7%9B%B8%E5%85%B3/CherryTree)编号为1-14
   >    > 第三个数字是该文档的数据集中，问答对的序号，从1开始
   > 3. 合并时存在重复问题时，仅保留一个，对于非重要问题，采取**忽略**的方式 *即也放入合并后的数据集*
4. 将合并后的数据集检查后上传至仓库
5. 当前阶段分文档进行转换，后续再将各文档的数据集进行合并
