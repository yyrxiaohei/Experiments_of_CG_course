## 学生信息

202411998117 闫一然 计算机科学与技术

# 实验六：可微渲染

## 1 项目结构

```text
Work6_Differentiable_Rendering/
├── assets/
│   └── Work6/
│       └── cow.obj
├── outputs/
│   └── Work6/
├── src/
│   └── Work6/
│       ├── __init__.py
│       ├── config.py
│       ├── main.py
│       ├── mesh_utils.py
│       ├── optimizer.py
│       ├── renderer.py
│       └── visualization.py
├── README.md
└── requirements.txt
```

主要文件说明：

- `config.py`：统一管理路径和实验参数。
- `mesh_utils.py`：负责模型加载、归一化、球体创建和模型保存。
- `renderer.py`：负责相机、软光栅化器和剪影渲染。
- `optimizer.py`：负责损失计算和顶点优化。
- `visualization.py`：负责保存中间结果和损失曲线。
- `main.py`：实验主入口。

## 2 实验原理

传统光栅化在三角形边界处不可导，难以将图像误差反向传播到网格顶点。软光栅化通过连续的边界概率近似，使剪影误差能够对顶点位置求导。

本实验从多个视角渲染目标奶牛模型的剪影，并优化球体顶点，使预测剪影逐步接近目标剪影。

为防止网格出现尖刺、拉伸和折叠，损失函数中加入：

- 拉普拉斯平滑损失；
- 边长损失；
- 法线一致性损失。

总损失为：

```text
L_total =
    L_silhouette
    + 1.0 * L_laplacian
    + 0.1 * L_edge
    + 0.01 * L_normal
```

## 3 模型准备

将教师提供的 `cow.obj` 放入：

```text
assets/Work6/cow.obj
```

## 运行方式

在项目根目录执行：

```bash
python -m src.Work6.main
```

使用 `uv` 时执行：

```bash
uv run -m src.Work6.main
```

## 4 运行流程

程序会依次完成：

1. 加载并归一化目标奶牛模型；
2. 创建多视角相机和软剪影渲染器；
3. 渲染目标奶牛的多视角剪影；
4. 创建细分球体作为初始模型；
5. 优化球体顶点偏移量；
6. 保存中间模型、最终模型和损失结果。

## 5 输出结果

运行结果保存在：

```text
outputs/Work6/
```

主要输出包括：

```text
outputs/Work6/
├── images/
├── output_meshes/
│   ├── source_sphere.obj
│   ├── target_cow_normalized.obj
│   ├── mesh_epoch_XXX.obj
│   └── final_optimized_mesh.obj
├── loss_curve.png
└── work6_optimization.gif
```

其中：

- `source_sphere.obj`：初始球体；
- `mesh_epoch_XXX.obj`：中间优化结果；
- `final_optimized_mesh.obj`：最终模型；
- `loss_curve.png`：损失变化曲线。

OBJ 文件可使用 MeshLab 或 Blender 查看。

## 6 结果说明

多视角剪影能够约束目标模型的整体三维轮廓，但无法仅依靠黑白剪影恢复颜色、纹理和所有几何细节。

网格正则化能够有效减轻顶点过度拉伸和局部尖刺，使优化结果更加稳定和平滑。
