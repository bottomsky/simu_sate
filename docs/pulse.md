ECI坐标系下施加脉冲后轨道根数计算（含平近点角）
根据您的要求，我将实现一个完整的轨道脉冲方案，在ECI坐标系下施加脉冲后计算新的轨道根数，特别关注平近点角的计算。
完整实现方案
1. 数据结构定义
#include <cmath>
#include <iostream>
#include <limits>

// 三维向量类
class Vector3 {
public:
    double x, y, z;
    
    Vector3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    
    double norm() const {
        return std::sqrt(x*x + y*y + z*z);
    }
    
    double norm_squared() const {
        return x*x + y*y + z*z;
    }
    
    Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }
    
    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }
    
    Vector3 operator*(double scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }
    
    Vector3 operator/(double scalar) const {
        return Vector3(x / scalar, y / scalar, z / scalar);
    }
    
    double dot(const Vector3& other) const {
        return x*other.x + y*other.y + z*other.z;
    }
    
    Vector3 cross(const Vector3& other) const {
        return Vector3(
            y*other.z - z*other.y,
            z*other.x - x*other.z,
            x*other.y - y*other.x
        );
    }
    
    Vector3 normalized() const {
        double n = norm();
        return n > 0 ? (*this / n) : Vector3(0, 0, 0);
    }
};

// 轨道根数结构体（含平近点角）
struct OrbitalElements {
    double a;       // 半长轴 (m)
    double e;       // 偏心率
    double i;       // 倾角 (rad)
    double Omega;   // 升交点赤经 (rad)
    double omega;   // 近地点幅角 (rad)
    double M;       // 平近点角 (rad)
};

// ECI坐标结构体
struct EciState {
    Vector3 r;  // 位置矢量 (m)
    Vector3 v;  // 速度矢量 (m/s)
};

2. 常数定义
// 地球引力常数 (m³/s²)
const double MU = 3.986004418e14;
// 地球赤道半径 (m)
const double EARTH_RADIUS = 6378137.0;
// ECI坐标系z轴单位矢量
const Vector3 K(0, 0, 1);
// π值
const double PI = 3.14159265358979323846;

3. ECI到轨道根数转换（含平近点角）
OrbitalElements eci_to_orbital_elements(const EciState& state) {
    OrbitalElements oe;
    const Vector3& r = state.r;
    const Vector3& v = state.v;
    
    // 计算基本物理量
    double r_norm = r.norm();
    double v_norm = v.norm();
    Vector3 h = r.cross(v);
    double h_norm = h.norm();
    double eps = 0.5 * v_norm * v_norm - MU / r_norm;
    
    // 计算半长轴
    if (std::fabs(eps) > 1e-9) {
        oe.a = -MU / (2 * eps);
    } else {
        oe.a = std::numeric_limits<double>::infinity();
    }
    
    // 计算偏心率矢量和偏心率
    Vector3 e_vec = ((v_norm * v_norm - MU / r_norm) * r - r.dot(v) * v) / MU;
    oe.e = e_vec.norm();
    
    // 计算倾角
    oe.i = std::acos(h.z / h_norm);
    // 处理数值误差
    if (oe.i < 0) oe.i = 0;
    if (oe.i > PI) oe.i = PI;
    
    // 计算节点矢量
    Vector3 n = K.cross(h);
    double n_norm = n.norm();
    
    // 计算升交点赤经
    if (n_norm < 1e-9) {
        oe.Omega = 0.0;
    } else {
        oe.Omega = std::atan2(n.y, n.x);
        if (oe.Omega < 0) oe.Omega += 2 * PI;
    }
    
    // 计算近地点幅角
    if (oe.e < 1e-9) {
        oe.omega = 0.0;
    } else if (n_norm < 1e-9) {
        oe.omega = std::atan2(e_vec.y, e_vec.x);
        if (oe.omega < 0) oe.omega += 2 * PI;
    } else {
        double cos_omega = e_vec.dot(n) / (oe.e * n_norm);
        cos_omega = std::max(std::min(cos_omega, 1.0), -1.0);
        oe.omega = std::acos(cos_omega);
        if (e_vec.dot(h) < 0) {
            oe.omega = 2 * PI - oe.omega;
        }
    }
    
    // 计算真近点角
    double nu = 0.0;
    if (oe.e < 1e-9) {
        nu = 0.0;
    } else {
        double cos_nu = e_vec.dot(r) / (oe.e * r_norm);
        cos_nu = std::max(std::min(cos_nu, 1.0), -1.0);
        nu = std::acos(cos_nu);
        if (r.dot(v) > 0) {
            nu = 2 * PI - nu;
        }
    }
    
    // 计算平近点角（关键部分）
    if (oe.e < 1e-9) {
        // 圆轨道：平近点角等于真近点角
        oe.M = nu;
    } else {
        // 计算偏近点角E
        double cosE = (oe.e + std::cos(nu)) / (1 + oe.e * std::cos(nu));
        double sinE = std::sqrt(1 - oe.e * oe.e) * std::sin(nu) / (1 + oe.e * std::cos(nu));
        double E = std::atan2(sinE, cosE);
        if (E < 0) E += 2 * PI;
        
        // 计算平近点角M
        oe.M = E - oe.e * std::sin(E);
        if (oe.M < 0) oe.M += 2 * PI;
        if (oe.M >= 2 * PI) oe.M -= 2 * PI;
    }
    
    return oe;
}

4. 施加脉冲函数
EciState apply_impulse(const EciState& state, const Vector3& delta_v) {
    EciState new_state = state;
    new_state.v = new_state.v + delta_v;
    return new_state;
}

5. 完整示例
int main() {
    // 示例初始ECI状态
    EciState initial_state;
    initial_state.r = Vector3(7.0e6, 0.0, 0.0);  // 位置矢量 (m)
    initial_state.v = Vector3(0.0, 7.5e3, 0.0);   // 速度矢量 (m/s)
    
    // 计算初始轨道根数
    OrbitalElements initial_oe = eci_to_orbital_elements(initial_state);
    
    // 输出初始轨道根数
    std::cout << "初始轨道根数：" << std::endl;
    std::cout << "半长轴 a = " << initial_oe.a / 1000 << " km" << std::endl;
    std::cout << "偏心率 e = " << initial_oe.e << std::endl;
    std::cout << "倾角 i = " << initial_oe.i * 180 / PI << " 度" << std::endl;
    std::cout << "升交点赤经 Ω = " << initial_oe.Omega * 180 / PI << " 度" << std::endl;
    std::cout << "近地点幅角 ω = " << initial_oe.omega * 180 / PI << " 度" << std::endl;
    std::cout << "平近点角 M = " << initial_oe.M * 180 / PI << " 度" << std::endl;
    
    // 定义脉冲：在速度方向施加100 m/s的脉冲
    Vector3 delta_v = initial_state.v.normalized() * 100.0;
    
    // 施加脉冲
    EciState new_state = apply_impulse(initial_state, delta_v);
    
    // 计算新的轨道根数
    OrbitalElements new_oe = eci_to_orbital_elements(new_state);
    
    // 输出新的轨道根数
    std::cout << "\n施加脉冲后的新轨道根数：" << std::endl;
    std::cout << "半长轴 a = " << new_oe.a / 1000 << " km" << std::endl;
    std::cout << "偏心率 e = " << new_oe.e << std::endl;
    std::cout << "倾角 i = " << new_oe.i * 180 / PI << " 度" << std::endl;
    std::cout << "升交点赤经 Ω = " << new_oe.Omega * 180 / PI << " 度" << std::endl;
    std::cout << "近地点幅角 ω = " << new_oe.omega * 180 / PI << " 度" << std::endl;
    std::cout << "平近点角 M = " << new_oe.M * 180 / PI << " 度" << std::endl;
    
    return 0;
}

关键点说明


平近点角计算流程：

首先计算真近点角ν
通过ν计算偏近点角E
使用开普勒方程 M = E - e·sin(E) 计算平近点角
确保平近点角在[0, 2π)范围内



**数值稳定性处理

