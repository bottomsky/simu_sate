# ECI 坐标系下施加脉冲后轨道根数计算指南（含平近点角及 C 格式结构体）

## 一、文档概述

本文档面向航天仿真工程师，旨在提供在地球中心惯性坐标系（ECI）下，从笛卡尔坐标推导包含平近点角的轨道根数、施加脉冲以及计算新轨道根数的完整实现方法。文档重点关注代码实现的可操作性，采用 C 格式的轨道要素结构体，提供详细的数学推导过程和 C 代码示例，确保工程师能够在实际工作中顺利实现相关功能。

## 二、数学基础与理论推导

### 2.1 轨道根数定义

在 ECI 坐标系中，卫星的轨道状态可以由六个轨道根数（或轨道要素）及历元时间唯一确定，C 格式的轨道要素结构体定义如下：



```
// C格式的轨道要素结构体

typedef struct {

&#x20;   double a;   // 半长轴 (m)

&#x20;   double e;   // 偏心率

&#x20;   double i;   // 倾角 (rad)

&#x20;   double O;   // 升交点赤经 (rad)

&#x20;   double w;   // 近地点幅角 (rad)

&#x20;   double M;   // 平近点角 (rad)

&#x20;   double t;   // 历元时间 (s)

} OrbitalElements;
```

各参数含义：



1.  **半长轴（a）**：轨道椭圆长轴的一半，决定了轨道的大小。

2.  **偏心率（e）**：描述轨道形状，0≤e<1 为椭圆轨道，e=0 为圆轨道。

3.  **倾角（i）**：轨道平面与赤道平面的夹角。

4.  **升交点赤经（O）**：从春分点到升交点的角度，升交点是卫星从南向北穿过赤道的点。

5.  **近地点幅角（w）**：从升交点到轨道近地点的角度。

6.  **平近点角（M）**：在椭圆轨道中，平近点角是一个与时间相关的角度参数，它均匀变化，用于轨道外推等计算。平近点角定义为：M = M₀ + n (t - t₀)，其中 M₀是历元时刻 t₀的平近点角，n 是平均角速度，n=√(μ/a³)，μ 是地球引力常数。

7.  **历元时间（t）**：轨道要素对应的参考时间。

这些轨道根数提供了比笛卡尔坐标更直观的轨道特性描述，在轨道预测、控制和分析中具有重要作用。

### 2.2 从笛卡尔坐标推导轨道根数

在 ECI 坐标系下，已知卫星的位置矢量 r 和速度矢量 v，可通过以下步骤计算包含平近点角的轨道根数：

#### 2.2.1 计算基本物理量

首先计算几个关键物理量：



1.  **角动量矢量（h）**：

    h = r × v

    角动量的大小为：

    h = ||h||

2.  **比机械能（ε）**：

    ε = v²/2 - μ/r

    其中，v 是速度矢量的大小，r 是位置矢量的大小，μ 是地球引力常数。

3.  **偏心率矢量（e）**：

    e = (1/μ)・\[(v² - μ/r)・r - (r・v)・v]

    偏心率的大小为：

    e = ||e||

4.  **平均角速度（n）**：

    n=√(μ/a³)，在计算出半长轴 a 后可得到平均角速度 n。

#### 2.2.2 推导轨道根数

基于上述物理量，按以下步骤计算各轨道根数：



1.  **半长轴（a）**：

    a = -μ/(2ε)

    该公式适用于椭圆轨道，对于抛物线或双曲线轨道需特殊处理。

2.  **倾角（i）**：

    i = arccos (h\_z /h)

    其中，h\_z 是角动量矢量在 ECI 坐标系 z 轴上的分量。

3.  **升交点赤经（O）**：

    O = arctan2 (n\_y, n\_x)

    其中，n 是节点矢量，定义为：

    n = k × h

    这里 k 是 ECI 坐标系 z 轴的单位矢量。

4.  **近地点幅角（w）**：

    w = arccos \[(n・e)/(n・e)]

    其中，n 是节点矢量的大小。需要根据 e 和 h 的关系调整象限。

5.  **平近点角（M）**：

    先通过真近点角 ν 计算偏近点角 E，利用公式 cosE = (e + cosν)/(1 + ecosν)。

    然后根据开普勒方程 M = E - e・sinE 计算平近点角 M。

真近点角 ν 计算方式如下：

ν = arccos \[(e・r)/(e・r)]

需要根据 r 和 v 的点积调整象限。

## 三、ECI 坐标系下的脉冲添加

### 3.1 脉冲机动原理

脉冲机动是通过在短时间内施加冲量，实现卫星轨道状态的突变。在 ECI 坐标系下，脉冲表现为速度矢量的增量 ΔV。脉冲机动的关键参数是速度增量的大小和方向，这决定了轨道变化的幅度和性质。

### 3.2 脉冲方向与轨道调整关系

在 ECI 坐标系下，不同方向的脉冲会对轨道产生不同的影响：



1.  **切向脉冲**：与当前速度矢量方向相同或相反的脉冲，主要影响半长轴和偏心率。

2.  **法向脉冲**：垂直于轨道平面的脉冲，主要影响倾角和升交点赤经。

3.  **径向脉冲**：沿位置矢量方向的脉冲，主要影响偏心率和近地点幅角。

### 3.3 脉冲添加的数学表达

在 ECI 坐标系下，施加脉冲后的新速度矢量为：

v\_new = v\_old + ΔV

位置矢量在脉冲瞬间保持不变：

r\_new = r\_old

因此，脉冲添加的数学实现非常直观，只需在速度矢量上加上 ΔV 即可。

## 四、施加脉冲后的轨道根数计算

### 4.1 新轨道根数推导流程

在 ECI 坐标系下施加脉冲后，新的轨道根数可以通过以下步骤计算：



1.  **更新速度矢量**：根据脉冲 ΔV 更新速度矢量。

2.  **计算新的基本物理量**：使用更新后的位置和速度矢量重新计算角动量、比机械能和偏心率矢量。

3.  **推导新的轨道根数**：基于新的基本物理量，按照 2.2 节所述方法计算新的轨道根数，特别注意平近点角的更新。由于平近点角与时间相关，在施加脉冲瞬间，时间发生了突变，需要重新计算平近点角。假设脉冲施加时刻为 t₁，施加脉冲前的平近点角为 M₀，平均角速度为 n，则新的平近点角 M = M₀ + n (t₁ - t₀)，同时更新历元时间为 t₁。

### 4.2 特殊情况处理

在计算新轨道根数时，需要特别注意以下特殊情况：



1.  **圆轨道（e≈0）**：当偏心率非常小时，近地点幅角和真近点角失去物理意义，近地点幅角通常设为 0。对于平近点角，其计算方式依然遵循 M = M₀ + n (t - t₀)，但此时平均角速度 n=√(μ/a³)，a 为圆轨道半径。

2.  **赤道轨道（i≈0 或 π）**：当倾角接近 0 或 π 时，升交点赤经失去意义，通常设为 0，近地点幅角直接基于 ECI 坐标系 x 轴计算。平近点角计算不受影响。

3.  **抛物线轨道（ε=0）**：此时半长轴趋向无穷大，平均角速度 n 的计算不再适用常规公式，平近点角的概念在抛物线轨道中也有所不同，需要特殊处理，通常可根据轨道的能量和角动量等参数重新定义相关角度参数用于轨道外推。

### 4.3 象限修正

在计算角度参数（O、w、ν）时，必须进行象限修正以确保结果的正确性：



1.  **O 的象限修正**：使用 atan2 函数自动处理，确保结果在 0 到 2π 之间。

2.  **w 的象限修正**：如果 e 和 h 的点积为负，说明计算出的 w 位于错误的象限，应取补角。

3.  **ν 的象限修正**：如果 r 和 v 的点积为正，说明卫星正在远离近地点，真近点角应取补角。平近点角本身是基于时间均匀变化的角度，不存在类似的象限修正问题，但在通过真近点角等推导过程中涉及的角度象限问题会影响到平近点角的最终计算准确性。

## 五、C 代码实现

### 5.1 数据结构定义

定义必要的数据结构来表示轨道状态和 ECI 坐标：



```
\#include \<math.h>

\#include \<limits.h>

// 三维向量结构体

typedef struct {

&#x20;   double x, y, z;

} Vector3;

// C格式的轨道要素结构体

typedef struct {

&#x20;   double a;   // 半长轴 (m)

&#x20;   double e;   // 偏心率

&#x20;   double i;   // 倾角 (rad)

&#x20;   double O;   // 升交点赤经 (rad)

&#x20;   double w;   // 近地点幅角 (rad)

&#x20;   double M;   // 平近点角 (rad)

&#x20;   double t;   // 历元时间 (s)

} OrbitalElements;

// ECI坐标结构体

typedef struct {

&#x20;   Vector3 r;  // 位置矢量 (m)

&#x20;   Vector3 v;  // 速度矢量 (m/s)

} EciState;

// 三维向量点积计算

double vector3\_dot(const Vector3\* v1, const Vector3\* v2) {

&#x20;   return v1->x \* v2->x + v1->y \* v2->y + v1->z \* v2->z;

}

// 三维向量叉积计算

Vector3 vector3\_cross(const Vector3\* v1, const Vector3\* v2) {

&#x20;   Vector3 result;

&#x20;   result.x = v1->y \* v2->z - v1->z \* v2->y;

&#x20;   result.y = v1->z \* v2->x - v1->x \* v2->z;

&#x20;   result.z = v1->x \* v2->y - v1->y \* v2->x;

&#x20;   return result;

}

// 三维向量模长计算

double vector3\_norm(const Vector3\* v) {

&#x20;   return sqrt(v->x \* v->x + v->y \* v->y + v->z \* v->z);

}

// 三维向量归一化

Vector3 vector3\_normalized(const Vector3\* v) {

&#x20;   Vector3 result;

&#x20;   double n = vector3\_norm(v);

&#x20;   if (n > 0) {

&#x20;       result.x = v->x / n;

&#x20;       result.y = v->y / n;

&#x20;       result.z = v->z / n;

&#x20;   } else {

&#x20;       result.x = 0;

&#x20;       result.y = 0;

&#x20;       result.z = 0;

&#x20;   }

&#x20;   return result;

}

// 三维向量加法

Vector3 vector3\_add(const Vector3\* v1, const Vector3\* v2) {

&#x20;   Vector3 result;

&#x20;   result.x = v1->x + v2->x;

&#x20;   result.y = v1->y + v2->y;

&#x20;   result.z = v1->z + v2->z;

&#x20;   return result;

}

// 三维向量数乘

Vector3 vector3\_multiply\_scalar(const Vector3\* v, double scalar) {

&#x20;   Vector3 result;

&#x20;   result.x = v->x \* scalar;

&#x20;   result.y = v->y \* scalar;

&#x20;   result.z = v->z \* scalar;

&#x20;   return result;

}
```

### 5.2 轨道根数计算函数

实现从 ECI 坐标到包含平近点角的轨道根数的转换函数：



```
// 地球引力常数 (m³/s²)

const double MU = 3.986004418e14;

// ECI坐标系z轴单位矢量

const Vector3 K = {0, 0, 1};

OrbitalElements eci\_to\_orbital\_elements(const EciState\* state, double t0, double M0) {

&#x20;   OrbitalElements oe;

&#x20;   const Vector3\* r = \&state->r;

&#x20;   const Vector3\* v = \&state->v;

&#x20;  &#x20;

&#x20;   // 计算基本物理量

&#x20;   double r\_norm = vector3\_norm(r);

&#x20;   double v\_norm = vector3\_norm(v);

&#x20;   Vector3 h = vector3\_cross(r, v);

&#x20;   double h\_norm = vector3\_norm(\&h);

&#x20;   double eps = 0.5 \* v\_norm \* v\_norm - MU / r\_norm;

&#x20;  &#x20;

&#x20;   // 计算半长轴

&#x20;   if (fabs(eps) > 1e-9) { // 避免除以零

&#x20;       oe.a = -MU / (2 \* eps);

&#x20;   } else {

&#x20;       oe.a = INFINITY; // 抛物线轨道

&#x20;   }

&#x20;  &#x20;

&#x20;   // 计算偏心率矢量和偏心率

&#x20;   double v\_squared = v\_norm \* v\_norm;

&#x20;   double term1 = v\_squared - MU / r\_norm;

&#x20;   Vector3 r\_scaled = vector3\_multiply\_scalar(r, term1);

&#x20;   double r\_dot\_v = vector3\_dot(r, v);

&#x20;   Vector3 v\_scaled = vector3\_multiply\_scalar(v, r\_dot\_v);

&#x20;   Vector3 e\_vec = vector3\_add(\&r\_scaled, \&vector3\_multiply\_scalar(\&v\_scaled, -1));

&#x20;   e\_vec = vector3\_multiply\_scalar(\&e\_vec, 1 / MU);

&#x20;   oe.e = vector3\_norm(\&e\_vec);

&#x20;  &#x20;

&#x20;   // 计算倾角

&#x20;   oe.i = acos(h.z / h\_norm);

&#x20;   // 处理数值误差导致的acos输入略超出\[-1,1]的情况

&#x20;   if (oe.i < 0) oe.i = 0;

&#x20;   if (oe.i > M\_PI) oe.i = M\_PI;

&#x20;  &#x20;

&#x20;   // 计算节点矢量

&#x20;   Vector3 n = vector3\_cross(\&K, \&h);

&#x20;   double n\_norm = vector3\_norm(\&n);

&#x20;  &#x20;

&#x20;   // 计算升交点赤经

&#x20;   if (n\_norm < 1e-9) { // 赤道轨道

&#x20;       oe.O = 0.0;

&#x20;   } else {

&#x20;       oe.O = atan2(n.y, n.x);

&#x20;       if (oe.O < 0) oe.O += 2 \* M\_PI; // 确保在\[0, 2π)范围内

&#x20;   }

&#x20;  &#x20;

&#x20;   // 计算近地点幅角

&#x20;   if (oe.e < 1e-9) { // 圆轨道

&#x20;       oe.w = 0.0;

&#x20;   } else if (n\_norm < 1e-9) { // 赤道轨道但非圆轨道

&#x20;       oe.w = atan2(e\_vec.y, e\_vec.x);

&#x20;       if (oe.w < 0) oe.w += 2 \* M\_PI;

&#x20;   } else {

&#x20;       double cos\_w = vector3\_dot(\&n, \&e\_vec) / (n\_norm \* oe.e);

&#x20;       // 处理数值误差导致的acos输入略超出\[-1,1]的情况

&#x20;       if (cos\_w > 1.0) cos\_w = 1.0;

&#x20;       if (cos\_w < -1.0) cos\_w = -1.0;

&#x20;       oe.w = acos(cos\_w);

&#x20;       // 象限修正：如果e\_vec和h的点积为负，说明计算出的w在错误的象限

&#x20;       if (vector3\_dot(\&e\_vec, \&h) < 0) {

&#x20;           oe.w = 2 \* M\_PI - oe.w;

&#x20;       }

&#x20;   }

&#x20;  &#x20;

&#x20;   // 计算真近点角（用于后续计算平近点角）

&#x20;   double nu;

&#x20;   if (oe.e < 1e-9) { // 圆轨道

&#x20;       nu = 0.0;

&#x20;   } else {

&#x20;       double cos\_nu = vector3\_dot(\&e\_vec, r) / (oe.e \* r\_norm);

&#x20;       // 处理数值误差导致的acos输入略超出\[-1,1]的情况

&#x20;       if (cos\_nu > 1.0) cos\_nu = 1.0;

&#x20;       if (cos\_nu < -1.0) cos\_nu = -1.0;

&#x20;       nu = acos(cos\_nu);

&#x20;       // 象限修正：如果r和v的点积为正，说明卫星正在远离近地点，真近点角应取补角

&#x20;       if (vector3\_dot(r, v) > 0) {

&#x20;           nu = 2 \* M\_PI - nu;

&#x20;       }

&#x20;   }

&#x20;  &#x20;

&#x20;   // 计算偏近点角E

&#x20;   double E = acos((oe.e + cos(nu)) / (1 + oe.e \* cos(nu)));

&#x20;  &#x20;

&#x20;   // 计算平近点角M

&#x20;   double n = sqrt(MU / (oe.a \* oe.a \* oe.a));

&#x20;   double t1 = 10.0; // 示例当前时刻，实际应根据具体情况获取

&#x20;   oe.M = M0 + n \* (t1 - t0\</doubaocanvas>
```

> （注：文档部分内容可能由 AI 生成）