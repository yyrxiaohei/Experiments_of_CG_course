# src/Work7/physics.py

import taichi as ti

from .config import *


# ====== 1 质点状态场 ======
positions = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
velocities = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
forces = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)

is_fixed = ti.field(dtype=ti.i32, shape=NUM_PARTICLES)


# ====== 2 隐式欧拉预测场 ======
next_positions = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
next_velocities = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
next_forces = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)


# ====== 3 弹簧数据场 ======
spring_pairs = ti.Vector.field(2, dtype=ti.i32, shape=MAX_SPRINGS)
spring_rest_lengths = ti.field(dtype=ti.f32, shape=MAX_SPRINGS)

# GGUI 绘制线段时，每条弹簧需要两个质点索引。
spring_indices = ti.field(dtype=ti.i32, shape=MAX_SPRINGS * 2)

num_springs = ti.field(dtype=ti.i32, shape=())


# ====== 4 分阶段初始化 Kernel ======
@ti.kernel
def init_positions():
    """初始化质点位置、速度、受力和固定状态。"""
    for i, j in ti.ndrange(CLOTH_RESOLUTION, CLOTH_RESOLUTION):
        idx = i * CLOTH_RESOLUTION + j

        positions[idx] = ti.Vector([
            i * CLOTH_SPACING + CLOTH_ORIGIN_X,
            CLOTH_INITIAL_HEIGHT,
            j * CLOTH_SPACING + CLOTH_ORIGIN_Z,
        ])

        velocities[idx] = ti.Vector([0.0, 0.0, 0.0])
        forces[idx] = ti.Vector([0.0, 0.0, 0.0])

        # 固定第一排的两个角点。
        if j == 0 and (i == 0 or i == CLOTH_RESOLUTION - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0


@ti.kernel
def init_springs():
    """初始化水平和竖直方向的结构弹簧。"""
    for i, j in ti.ndrange(CLOTH_RESOLUTION, CLOTH_RESOLUTION):
        idx = i * CLOTH_RESOLUTION + j

        # x 方向相邻质点。
        if i < CLOTH_RESOLUTION - 1:
            idx_right = (i + 1) * CLOTH_RESOLUTION + j

            spring_id = ti.atomic_add(num_springs[None], 1)
            spring_pairs[spring_id] = ti.Vector([idx, idx_right])
            spring_rest_lengths[spring_id] = (
                positions[idx] - positions[idx_right]
            ).norm()

        # z 方向相邻质点。
        if j < CLOTH_RESOLUTION - 1:
            idx_down = i * CLOTH_RESOLUTION + (j + 1)

            spring_id = ti.atomic_add(num_springs[None], 1)
            spring_pairs[spring_id] = ti.Vector([idx, idx_down])
            spring_rest_lengths[spring_id] = (
                positions[idx] - positions[idx_down]
            ).norm()


@ti.kernel
def init_spring_indices():
    """将弹簧端点同步到 GGUI 线段索引场。"""
    for spring_id in range(num_springs[None]):
        spring_indices[spring_id * 2] = spring_pairs[spring_id][0]
        spring_indices[spring_id * 2 + 1] = spring_pairs[spring_id][1]


def init_cloth():
    """从 Python 层依次调用初始化 Kernel，保证 GPU 状态同步。"""
    num_springs[None] = 0

    init_positions()
    init_springs()
    init_spring_indices()


# ====== 5 力学计算内联函数 ======
@ti.func
def compute_forces_on(
    pos: ti.template(),
    vel: ti.template(),
    force: ti.template(),
):
    """计算重力、阻尼力和结构弹簧力。"""
    gravity = ti.Vector([GRAVITY_X, GRAVITY_Y, GRAVITY_Z])

    # 第一阶段：清空并写入每个质点独立的重力与阻尼力。
    for i in range(NUM_PARTICLES):
        force[i] = gravity * PARTICLE_MASS - DAMPING_COEFFICIENT * vel[i]

    # 第二阶段：并行累加弹簧力。
    for spring_id in range(num_springs[None]):
        idx_a = spring_pairs[spring_id][0]
        idx_b = spring_pairs[spring_id][1]

        displacement = pos[idx_a] - pos[idx_b]
        distance = displacement.norm()

        if distance > DISTANCE_EPSILON:
            direction = displacement / distance

            spring_force = (
                -STRUCTURAL_STIFFNESS
                * (distance - spring_rest_lengths[spring_id])
                * direction
            )

            ti.atomic_add(force[idx_a], spring_force)
            ti.atomic_add(force[idx_b], -spring_force)


@ti.func
def clamp_velocity(vel: ti.template(), idx: ti.i32):
    """限制质点最大速度，防止数值爆炸。"""
    speed = vel[idx].norm()

    if speed > MAX_VELOCITY:
        vel[idx] = vel[idx] / speed * MAX_VELOCITY


# ====== 6 数值积分 Kernel ======
@ti.kernel
def step_explicit():
    """显式欧拉：先使用旧速度更新位置，再使用旧力更新速度。"""
    compute_forces_on(positions, velocities, forces)

    for i in range(NUM_PARTICLES):
        if is_fixed[i] == 0:
            positions[i] += velocities[i] * TIME_STEP
            velocities[i] += forces[i] / PARTICLE_MASS * TIME_STEP

            clamp_velocity(velocities, i)


@ti.kernel
def step_semi_implicit():
    """半隐式欧拉：先更新速度，再使用新速度更新位置。"""
    compute_forces_on(positions, velocities, forces)

    for i in range(NUM_PARTICLES):
        if is_fixed[i] == 0:
            velocities[i] += forces[i] / PARTICLE_MASS * TIME_STEP
            clamp_velocity(velocities, i)

            positions[i] += velocities[i] * TIME_STEP


@ti.kernel
def step_implicit_iter():
    """隐式欧拉：通过固定次数的定点迭代近似求解未来状态。"""
    # 1. 将当前状态复制到预测场。
    for i in range(NUM_PARTICLES):
        next_positions[i] = positions[i]
        next_velocities[i] = velocities[i]

    # 2. 编译期展开定点迭代。
    for _ in ti.static(range(IMPLICIT_ITERATIONS)):
        compute_forces_on(
            next_positions,
            next_velocities,
            next_forces,
        )

        for i in range(NUM_PARTICLES):
            if is_fixed[i] == 0:
                next_velocities[i] = (
                    velocities[i]
                    + next_forces[i] / PARTICLE_MASS * TIME_STEP
                )

                clamp_velocity(next_velocities, i)

                next_positions[i] = (
                    positions[i] + next_velocities[i] * TIME_STEP
                )

    # 3. 将预测结果写回当前状态。
    for i in range(NUM_PARTICLES):
        velocities[i] = next_velocities[i]
        positions[i] = next_positions[i]
