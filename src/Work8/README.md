## 学生信息

202411998117 闫一然 计算机科学与技术

# 实验八：LBS 蒙皮

## 1. 项目说明

本项目基于 PyTorch、SMPL 和 Matplotlib，手写复现 SMPL 的线性混合蒙皮（Linear Blend Skinning，LBS）流程，并可视化以下四个阶段：

1. 模板网格与蒙皮权重；
2. 形状校正与关节回归；
3. 姿态校正；
4. 最终线性混合蒙皮结果。

项目同时将手写 LBS 结果与 `smplx` 官方前向结果进行误差比较，并完成固定体型下的单关节姿态动画。

## 2. 项目结构

```text
CG_Lab/
├── pyproject.toml
└── src/
    └── Work8/
        ├── __init__.py
        ├── config.py
        ├── model_utils.py
        ├── lbs.py
        ├── visualization.py
        ├── animation.py
        ├── main.py
        ├── README.md
        ├── models/
        │   └── smpl/
        │       └── SMPL_NEUTRAL.pkl
        └── outputs/
```

各文件功能如下：

- `config.py`：集中管理模型、姿态、相机和动画参数；
- `model_utils.py`：负责模型路径处理、旧版 pickle 兼容和 SMPL 加载；
- `lbs.py`：实现形状混合、关节回归、姿态校正、刚体变换和手写 LBS；
- `visualization.py`：负责网格、骨架、权重和四阶段对比图绘制；
- `animation.py`：生成单关节旋转动画；
- `main.py`：程序入口，负责完整实验流程与结果保存。

其中，本项目不包含 `SMPL_NEUTRAL.pkl` 模型文件。需根据课程要求或 SMPL 官方许可自行获取模型，并将其放置在：

```text
src/Work8/models/smpl/SMPL_NEUTRAL.pkl
```

SMPL 模型文件受官方许可协议约束，不应提交到公开 Git 仓库，也不应随项目代码再次分发。仓库中仅保留模型目录与使用说明。

建议在项目根目录的 `.gitignore` 中加入：

```gitignore
src/Work8/models/smpl/*.pkl
src/Work8/models/smpl/*.npz
```

## 3. 环境准备

项目使用 `uv` 管理依赖。在项目根目录执行：

```bash
uv sync
```

将课程提供的 `SMPL_NEUTRAL.pkl` 放置到：

```text
src/Work8/models/smpl/SMPL_NEUTRAL.pkl
```

SMPL 模型文件受许可协议约束，不应提交到公开仓库。

## 4. 运行方法

执行必做与选做部分：

```bash
uv run -m src.Work8.main
```

仅执行必做部分：

```bash
uv run -m src.Work8.main --skip-animation
```

指定权重关节和动画参数：

```bash
uv run -m src.Work8.main   --joint-id 18   --animation-joint-id 18   --animation-axis y   --animation-angle -75
```

## 5. 核心实现

### 5.1 形状校正与关节回归

```python
v_shaped = v_template + blend_shapes(betas, shapedirs)
J = vertices2joints(J_regressor, v_shaped)
```

`v_shaped` 表示加入形状参数后的网格，`J` 表示从该网格回归得到的关节位置。

### 5.2 姿态校正

```python
rot_mats = batch_rodrigues(full_pose.reshape(-1, 3))
pose_feature = rot_mats[:, 1:] - identity
pose_offsets = pose_feature @ posedirs
v_posed = v_shaped + pose_offsets
```

该阶段加入姿态相关的非刚性几何修正，但尚未执行骨骼蒙皮。

### 5.3 线性混合蒙皮

```python
J_transformed, A = batch_rigid_transform(rot_mats, J, parents)
T = lbs_weights @ A
verts = T @ homogeneous(v_posed)
```

每个顶点使用蒙皮权重对多个关节变换进行加权，从而在关节附近形成平滑过渡。

## 6. 核心对象

| 变量 | 含义 |
|---|---|
| `v_template` | 原始 T-pose 模板顶点 |
| `v_shaped` | 加入形状校正后的顶点 |
| `J` | 从 `v_shaped` 回归得到的静止关节 |
| `v_posed` | 加入姿态校正但尚未蒙皮的顶点 |
| `verts` | 完成 LBS 后的最终顶点 |

## 7. 输出结果

程序运行后在 `src/Work8/outputs/` 中生成：

```text
stage_a_template_weights.png
all_joint_weights.png
stage_b_shaped_joints.png
stage_c_pose_offsets.png
stage_d_lbs_result.png
comparison_grid.png
summary.txt
animation_frames/
lbs_animation.gif
lbs_animation.mp4
```

其中：

- `stage_a_template_weights.png` 展示模板网格和指定关节的权重；
- `all_joint_weights.png` 展示各区域的主导关节；
- `stage_b_shaped_joints.png` 展示形状变化后的网格与回归关节；
- `stage_c_pose_offsets.png` 展示姿态校正的空间分布；
- `stage_d_lbs_result.png` 展示最终蒙皮姿态；
- `comparison_grid.png` 对比四个 LBS 阶段；
- `summary.txt` 记录模型信息和手写结果误差；
- GIF、MP4 和逐帧图片记录单关节动画结果。

## 8. 实现内容

- 加载 neutral SMPL 模型并输出基础信息；
- 可视化单关节权重和全关节主导权重；
- 计算 `v_shaped` 和回归关节 `J`；
- 计算 `pose_feature`、`pose_offsets` 和 `v_posed`；
- 手写运动学链刚体变换与线性混合蒙皮；
- 生成四阶段对比结果；
- 计算手写结果与官方前向结果的平均绝对误差、最大绝对误差、均方根误差和最大顶点 L2 误差；
- 固定 shape 参数并生成单关节旋转动画。

## 9. 实验结论

蒙皮权重使关节交界处的顶点能够同时受多个骨骼影响，从而避免分块刚性运动。形状参数不仅改变人体表面，也会改变合理的关节位置。姿态校正用于补偿纯刚体混合在肩、肘、髋和膝等弯曲区域产生的体积损失与不自然折痕。最终 LBS 通过加权组合多个关节变换，实现连续、平滑的人体姿态变形。
