# 接口文档（简要）

本文档仅记录与本次实施相关的 Python 绑定测试侧接口使用要点。

## J2OrbitPropagator（python_binding_example 导出）
- 构造函数 J2OrbitPropagator(initial: dict)
  - 参数：
    - initial: 初始轨道要素字典，键包含 a,e,i,O,w,M,t
  - 返回：
    - 实例对象
  - 异常：
    - 参数缺失或类型错误将抛异常
- set_step_size(step: float) -> None
  - 参数：步长（秒）
  - 返回：无
  - 异常：非法步长参数可能抛异常
- propagate(t: float) -> dict
  - 参数：目标时间（秒）
  - 返回：轨道要素字典（a,e,i,O,w,M,t）
  - 异常：底层库调用失败时抛异常
- apply_impulse(state: dict, dv: list[float], t_impulse: float) -> dict
  - 参数：
    - state: 参考状态根数
    - dv: 脉冲矢量 [dvx,dvy,dvz]
    - t_impulse: 施加时刻
  - 返回：施加脉冲后的轨道要素
  - 异常：
    - 底层符号缺失或调用失败抛异常
- set_adaptive_parameters(rtol: float, atol: float, dt_max: float) -> None
  - 参数：相对/绝对误差与最大步长
  - 返回：无
  - 异常：底层符号缺失或参数非法抛异常

## 辅助函数（测试侧）
- _angle_diff(a: float, b: float) -> float
  - 计算角度环域差（[-pi, pi]）。
- _almost_equal_elements(e1: dict, e2: dict) -> bool
  - 领域容差比较几何要素，忽略 t 字段。
- _save_json(path: str, data: dict) -> None
  - 保存 JSON（UTF-8）。