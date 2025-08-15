# J2OrbitPropagator 使用手册（轨道外推与 ECI 脉冲）

本手册面向需要在 C++ 中使用 J2 摄动轨道传播与脉冲（ΔV）应用的开发者，涵盖：
- 公共 API 说明（C++）
- 数据结构、单位与坐标系
- 快速开始（含完整 C++ 示例）
- 外推（propagate）、施加脉冲（applyImpulse）、元素/状态互转
- 固定步长与自适应步长配置
- 数值与物理注意事项、常见问题

对应源码位置：`include/j2_orbit_propagator.h` 与 `src/j2_orbit_propagator.cpp`。

---

## 1. 公共 API（C++）

- 构造与析构
  - `J2OrbitPropagator(const OrbitalElements& initial_elements)`：用初始轨道根数构造传播器。
- 轨道传播
  - `OrbitalElements propagate(double t)`：将当前轨道外推到目标时刻 t（秒）。
- 元素/状态互转
  - `StateVector elementsToState(const OrbitalElements& elements)`：从轨道根数计算 ECI 位置/速度。
  - `OrbitalElements stateToElements(const StateVector& state, double t)`：从 ECI 位置/速度计算轨道根数（历元设为 t）。
- 施加脉冲（ECI）
  - `OrbitalElements applyImpulse(const OrbitalElements& elements, const Eigen::Vector3d& delta_v, double t)`：在 ECI 系对速度施加 ΔV，返回新的轨道根数（历元为 t）。
- 步长与自适应参数（可选）
  - `void setStepSize(double step)` / `double getStepSize() const`
  - `void setAdaptiveStepSize(bool enable)`
  - `void setAdaptiveParameters(double tolerance = 1e-6, double min_step = 1.0, double max_step = 300.0)`

---

## 2. 数据结构与单位

- OrbitalElements（轨道根数）
  - `a` 半长轴（m）、`e` 偏心率（无量纲）、`i` 倾角（rad）、`O` 升交点赤经（rad）、`w` 近地点幅角（rad）、`M` 平近点角（rad）、`t` 历元时间（s）
- StateVector（ECI 状态）
  - `r` 位置向量（m），`v` 速度向量（m/s）
- 单位与坐标系
  - 角度单位为弧度，时间单位为秒，长度单位为米，速度单位为米/秒。
  - 施加的 ΔV 必须位于地心惯性系（ECI）。

---

## 3. 快速开始：完整 C++ 示例

下面给出一个最小可编译示例，演示：初始化 → 外推 → 施加脉冲 → 元素/状态互转 → 自适应步长设置。

```cpp
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include "j2_orbit_propagator.h"

static void printElements(const OrbitalElements& oe, const char* tag) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "[" << tag << "] t=" << oe.t
              << " a=" << oe.a
              << " e=" << oe.e
              << " i(rad)=" << oe.i
              << " O(rad)=" << oe.O
              << " w(rad)=" << oe.w
              << " M(rad)=" << oe.M << std::endl;
}

int main() {
    // 1) 定义初始轨道根数（LEO 近极轨的一个示例）
    OrbitalElements oe0{};
    oe0.a = 7000e3;               // 7000 km
    oe0.e = 0.001;                // 小偏心
    oe0.i = 98.0 * M_PI / 180.0;  // 倾角 98°（rad）
    oe0.O = 30.0 * M_PI / 180.0;
    oe0.w = 45.0 * M_PI / 180.0;
    oe0.M = 0.0;                  // 平近点角
    oe0.t = 0.0;                  // 历元（s）

    J2OrbitPropagator prop(oe0);

    // 可选：配置固定步长或自适应步长
    prop.setStepSize(60.0);                 // 固定步长 60s（默认）
    prop.setAdaptiveStepSize(true);         // 启用自适应
    prop.setAdaptiveParameters(1e-6, 1.0, 300.0); // 误差容限/最小/最大步长

    // 2) 外推到 t1
    double t1 = 3600.0; // 1 小时后
    OrbitalElements oe_t1 = prop.propagate(t1);
    printElements(oe_t1, "after propagate to t1");

    // 3) 在 t1 时刻施加 ΔV（ECI），例如沿 x 轴 0.1 m/s
    Eigen::Vector3d dv(0.1, 0.0, 0.0);
    OrbitalElements oe_after_impulse = prop.applyImpulse(oe_t1, dv, t1);
    printElements(oe_after_impulse, "after impulse at t1");

    // 4) 元素 -> 状态（ECI）
    StateVector s = prop.elementsToState(oe_after_impulse);
    std::cout << "r(m) = [" << s.r.x() << ", " << s.r.y() << ", " << s.r.z() << "]\n";
    std::cout << "v(m/s) = [" << s.v.x() << ", " << s.v.y() << ", " << s.v.z() << "]\n";

    // 5) 状态 -> 元素（历元设为 t2）
    double t2 = t1 + 600.0; // 10 分钟后
    OrbitalElements oe_from_state = prop.stateToElements(s, t2);
    printElements(oe_from_state, "state -> elements at t2");

    // 6) 继续从 t1 往后外推到 t2（可选择以 oe_after_impulse 为新的初始态）
    // 这里直接用同一个传播器继续 propagate（内部状态已在构造时归一化处理角度）
    OrbitalElements oe_t2 = prop.propagate(t2);
    printElements(oe_t2, "after propagate to t2");

    return 0;
}
```

编译时确保包含 Eigen 和本项目的 include 目录，例如（示例命令，具体以工程 CMake 配置为准）：
- 包含目录：`-I <project>/include -I <project>/lib/eigen`
- 链接库：若以静/动库形式集成，请按工程实际链接设置。

---

## 4. 外推 propagate 与脉冲 applyImpulse 的要点

- propagate(t)
  - 不支持时间倒推：t 必须 ≥ 当前历元时间，否则将直接返回当前状态并输出错误提示。
  - 固定步长：以 `step_size_` 为步长循环积分至目标时刻。
  - 自适应步长：结合误差估计自动缩放步长，使误差控制在 `tolerance_` 内，并受 `min_step_size_`/`max_step_size_` 约束。
- applyImpulse(elements, ΔV, t)
  - ΔV 为 ECI 坐标系下的速度增量，位置保持不变；返回的新根数的历元设置为 t。
  - 内部流程：元素→状态（ECI）→速度加 ΔV→状态→新元素（含正确的平近点角 M 与角度归一化）。

---

## 5. 固定步长与自适应步长配置建议

- 固定步长：
  - 优点：实现简单、结果稳定可复现。
  - 建议：LEO 典型值 30–60 s；较高轨道可适当增大。
- 自适应步长：
  - 通过 `setAdaptiveStepSize(true)` 启用，`setAdaptiveParameters(tol, min, max)` 配置。
  - 建议：`tol` 从 1e-6 起试验；`min` 设为 0.5–5 s，`max` 设为 120–600 s 依据工况调优。

---

## 6. 数值与物理注意事项

- 角度归一化：构造与转换阶段会将 i, O, w, M 归一化到 [0, 2π)。
- 轨道类型：本实现主要针对椭圆轨道；大 ΔV 可能导致轨道类型突变（近抛/双曲），在极端情况下需谨慎评估。
- 坐标系一致性：确保 ΔV 与由 `elementsToState` 得到的速度向量同处 ECI 坐标系。
- 连续传播：若需要以脉冲后的状态继续传播，可将返回的新根数作为新的初始条件，或按工程逻辑更新传播器内部状态。

---

## 7. 常见问题（FAQ）

- Q: 为什么 `applyImpulse` 不直接修改传播器内部的当前状态？
  - A: 该函数采用纯函数风格，输入元素与 ΔV、t，返回新元素。你可以按需要将其作为新的初始元素。
- Q: 如何获取 ECI 下的位置与速度？
  - A: 使用 `elementsToState` 完成元素→状态转换。
- Q: 如何切换固定步长与自适应步长？
  - A: 通过 `setAdaptiveStepSize(true/false)` 切换，并用 `setAdaptiveParameters` 设置误差与步长范围；`setStepSize` 控制固定步长。

---

如需 C 语言接口或跨语言绑定，请参考：
- C 头文件：`include/j2_orbit_propagator_c.h`
- 示例：`example/c_example.c`、`example/CSharpBindingExample.cs`、`example/python_binding_example.py`