`EA_data`和`EA_gaea`文件夹是entity alignment领域常用的数据格式，用三元组描述网络关系。

这两个文件夹的数据是符合Dual-AMN, GAEA和JMAC_EA三个实体对齐算法的输入格式。

其中，`EA_data`中两个网络的节点编号都从0开始，而在`EA_gaea`中，第二个网络的节点编号则是从第一个网络的最后一个节点编号之后开始。

而network alignment常使用邻接矩阵即`adj_s.pkl`和`adj_t.pkl`作为输入。