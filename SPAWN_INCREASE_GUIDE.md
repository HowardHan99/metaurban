# 动态行人递增功能 (Dynamic Pedestrian Spawn Increase)

## 概述

`spawn_increase_per_episode` 功能允许在每个 episode 的 reset 时自动增加行人数量。这在数据收集或训练场景中很有用，可以逐步增加环境复杂度。

## 配置参数

### `spawn_increase_per_episode` (默认: 0)

- **类型**: `int`
- **说明**: 每次 reset 增加的行人数量。设置为 0 表示不增加（保持常数）。
- **示例**:
  - `spawn_increase_per_episode=0`: 每个 episode 都是 40 人（如果 base=40）
  - `spawn_increase_per_episode=5`: 第1个 episode 40 人，第2个 45 人，第3个 50 人，...

## 使用方法

### 方法1: CLI 参数（推荐）

```bash
# 基础用法：起始 40 人，每个 episode 增加 5 人
python metaurban/examples/drive_social_with_pretrained_policy.py \
  --spawn-human-num 40 \
  --spawn-increase-per-episode 5

# 或者从 30 人开始，每个 episode 增加 10 人
python metaurban/examples/drive_social_with_pretrained_policy.py \
  --spawn-human-num 30 \
  --spawn-increase-per-episode 10

# 禁用递增（保持常数）
python metaurban/examples/drive_social_with_pretrained_policy.py \
  --spawn-human-num 50 \
  --spawn-increase-per-episode 0
```

### 方法2: 直接配置

```python
from metaurban.envs.social_dynamic_env import SocialDynamicMetaUrbanEnv

config = {
    "spawn_human_num": 40,
    "spawn_increase_per_episode": 5,
    # ... 其他配置
}

env = SocialDynamicMetaUrbanEnv(config)
for episode in range(10):
    obs, info = env.reset()
    # Episode 0: 40 人
    # Episode 1: 45 人
    # Episode 2: 50 人
    # ...
```

## 日志输出示例

```
Episode 0: dynamically set spawn_human_num=40 (base=40 + 0*5)
Episode 1: dynamically set spawn_human_num=45 (base=40 + 1*5)
Episode 2: dynamically set spawn_human_num=50 (base=40 + 2*5)
Episode 3: dynamically set spawn_human_num=55 (base=40 + 3*5)
```

## 数学公式

每个 episode 的行人数量计算如下：

```
spawn_human_num[episode] = base_spawn + episode_count * spawn_increase_per_episode
```

其中:
- `base_spawn`: 初始的 `spawn_human_num` 配置值
- `episode_count`: 从 0 开始计数的 episode 索引（0-indexed）
- `spawn_increase_per_episode`: 每个 episode 增加的数量

## 实现细节

1. **首次 reset 时保存 base 值**: 第一次调用 reset() 时，会保存 base spawn 值到 `_base_spawn_human_num`，后续 reset 都使用这个值计算。
2. **reset 计数器**: 每次成功 reset 后，内部计数器 `_reset_count` 会递增。
3. **配置更新**: 动态计算的 `spawn_human_num` 在 reset 前被更新到 config 中。

## 常用场景

### 场景1: 渐进式难度增加

```bash
# 从 30 人开始，每轮增加 5 人，模拟环境复杂度递增
python drive_social.py --spawn-human-num 30 --spawn-increase-per-episode 5
```

### 场景2: 数据多样性收集

```bash
# 收集不同密度下的数据：30, 40, 50, 60, 70, 80 人
python collect_dataset.py --spawn-human-num 30 --spawn-increase-per-episode 10
```

### 场景3: 固定行人数（不增加）

```bash
# 保持每个 episode 都是 50 人
python drive_social.py --spawn-human-num 50 --spawn-increase-per-episode 0
```

## 注意事项

- 行人总数不能超过环境能处理的最大值（通常由计算资源限制）
- 需要确保有足够的地图空间容纳增加的行人
- 建议逐步增加（如 +5 或 +10），避免突然跳跃导致行为不稳定
- 与 `group_release_enable` 配合使用时，确保行人池足够大以维持群体

## 相关参数

- `spawn_human_num`: 基础行人数量（第一个 episode 的行人数）
- `group_cluster_num`: 群体数量
- `crossing_ped_num`: 过马路的行人数
- `vulnerable_ped_num`: 易受伤行人数
