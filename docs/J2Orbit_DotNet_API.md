# J2Orbit .NET API 文档

命名空间：J2.Propagator  
单位约定：角度=弧度（除 LatDeg/LonDeg），长度=米，速度=米/秒，时间=相对 J2000.0 的秒（J2000.0=2000-01-01T12:00:00Z）。向量数组长度为 3。

---

## 异常
- NativeCallException(msg)：原生库调用失败的基类异常。
- PropagationException(msg)：传播/施加脉冲失败。
- ConversionException(msg)：元素/状态与坐标转换失败。

---

## 数据结构
### COrbitalElements（与原生 C 布局一致）
- a：半长轴（m）
- e：偏心率
- i：倾角（rad）
- O：升交点赤经 Ω（rad）
- w：近地点幅角 ω（rad）
- M：平近点角 M（rad）
- t：历元时间（J2000 秒）

### CStateVector（与原生 C 布局一致）
- r：ECI 位置（m，len=3）
- v：ECI 速度（m/s，len=3）
- Create()：初始化 r、v 为长度 3 的数组。

### GeodeticCoord（大地坐标）
- LatRad：纬度（rad）
- LonRad：经度（rad）
- AltMeters：高程（m）
- LatDeg/LonDeg：度制便捷属性
- FromDegrees(latDeg, lonDeg, altMeters)：度制构造。

---

## 时间工具：TimeUtil
- J2000EpochUtc：J2000.0（UTC）。
- ToSecondsSinceJ2000(utc)：UTC→J2000 秒。
- FromSecondsSinceJ2000(seconds)：J2000 秒→UTC。

---

## 坐标转换：GeoConversion
- EcefToGeodetic(x, y, z) → GeodeticCoord：ECEF→大地。
- GeodeticToEcef(geo) → (x, y, z)：大地→ECEF。

---

## 传播器封装：J2Orbit（IDisposable）
J2 摄动轨道传播器的 .NET 封装，提供传播、元素/状态转换、ECI/ECEF/大地坐标转换、GMST 计算等。

### 构造与释放
- J2Orbit(initial: COrbitalElements)
  - 失败抛出 NativeCallException。
- Dispose()：释放非托管资源（析构函数也会调用）。

### 传播
- Propagate(targetTime: double) → COrbitalElements
  - 输入：J2000 秒；失败抛出 PropagationException。
- PropagateUtc(utc: DateTime) → COrbitalElements
  - 以 UTC 指定时刻传播。

### 元素/状态互转（ECI）
- ElementsToState(elements) → CStateVector
  - 失败抛出 ConversionException。
- StateToElements(state, time: double) → COrbitalElements
  - time 为 J2000 秒；失败抛出 ConversionException。
- StateToElementsUtc(state, utc: DateTime) → COrbitalElements

### 瞬时脉冲（ECI）
- ApplyImpulse(elements, deltaV[3], t: double) → COrbitalElements
  - deltaV 长度必须为 3；失败抛出 PropagationException。
- ApplyImpulseUtc(elements, deltaV[3], utc: DateTime) → COrbitalElements

### 步长与自适应
- StepSize { get; set; }（秒）
  - 失败抛出 NativeCallException。
- SetAdaptive(enable: bool)
- SetAdaptiveParameters(tol, minStep, maxStep)

### 角度与多坐标封装
- NormalizeAngle(angle) → double：归一化到 (-π, π]。
- EciToEcefPosition(rEci[3], utc) → rEcef[3]
- EcefToEciPosition(rEcef[3], utc) → rEci[3]
- EciToEcefVelocity(rEci[3], vEci[3], utc) → vEcef[3]
- EcefToEciVelocity(rEcef[3], vEcef[3], utc) → vEci[3]
- EciToGeodetic(rEci[3], utc) → GeodeticCoord
- GeodeticToEci(geo, utc) → rEci[3]
- ComputeGmstUtc(utc, out gmstRad) → int（0 成功）

---

## 示例

创建传播器并传播到指定 UTC：
```csharp
using J2.Propagator;

var el0 = new COrbitalElements {
    a = 7000e3, e = 0.001, i = 98 * Math.PI/180.0,
    O = 0, w = 0, M = 0, t = 0
};

using var j2 = new J2Orbit(el0);
var elNow = j2.PropagateUtc(DateTime.UtcNow);
var sv = j2.ElementsToState(elNow);
var geo = J2Orbit.EciToGeodetic(sv.r, DateTime.UtcNow);
J2Orbit.ComputeGmstUtc(DateTime.UtcNow, out var gmst);
```

ECI/ECEF 与大地坐标互转：
```csharp
var geoDeg = GeodeticCoord.FromDegrees(30, 114, 100);
var utc = DateTime.UtcNow;
var rEci = J2Orbit.GeodeticToEci(geoDeg, utc);
var rEcef = J2Orbit.EciToEcefPosition(rEci, utc);
var geo2 = GeoConversion.EcefToGeodetic(rEcef[0], rEcef[1], rEcef[2]);
```

施加瞬时脉冲（ECI）：
```csharp
var dv = new[] {0.0, 0.5, 0.0}; // m/s
var el1 = j2.ApplyImpulseUtc(el0, dv, DateTime.UtcNow);
```

---

提示：所有角度参数均为弧度（除 LatDeg/LonDeg）；数组参数长度必须为 3；UTC 输入会自动转换为 UTC 时区。