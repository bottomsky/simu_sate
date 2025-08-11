#include "j2_orbit_propagator.h"

J2OrbitPropagator::J2OrbitPropagator(const OrbitalElements& initial_elements) 
    : current_elements_(initial_elements), step_size_(10.0) {
    // 确保角度参数在正确范围内
    current_elements_.i = normalizeAngle(current_elements_.i);
    current_elements_.O = normalizeAngle(current_elements_.O);
    current_elements_.w = normalizeAngle(current_elements_.w);
    current_elements_.M = normalizeAngle(current_elements_.M);
}

OrbitalElements J2OrbitPropagator::propagate(double t) {
    // 计算需要外推的时间差
    double dt_total = t - current_elements_.t;
    
    if (dt_total < 0) {
        std::cerr << "Error: Target time is earlier than epoch time." << std::endl;
        return current_elements_;
    }
    
    // 如果时间差很小，直接返回当前要素
    if (dt_total < EPSILON) {
        return current_elements_;
    }
    
    // 保存初始时间
    double initial_time = current_elements_.t;
    
    // 分步骤积分
    double remaining_time = dt_total;
    while (remaining_time > EPSILON) {
        double dt = std::min(remaining_time, step_size_);
        current_elements_ = rk4Integrate(current_elements_, dt);
        current_elements_.t += dt;
        remaining_time -= dt;
    }
    
    // 更新历元时间
    current_elements_.t = t;
    
    return current_elements_;
}

Eigen::VectorXd J2OrbitPropagator::computeDerivatives(const OrbitalElements& elements) {
    // 提取轨道要素
    double a = elements.a;
    double e = elements.e;
    double i = elements.i;
    double O = elements.O;
    double w = elements.w;
    double M = elements.M;
    
    // 防止偏心率为1导致的数值问题
    if (std::abs(e - 1.0) < EPSILON) {
        return Eigen::VectorXd::Zero(6);
    }
    
    // 计算辅助参数
    double n = std::sqrt(MU / std::pow(a, 3));  // 平均角速度
    double p = a * (1.0 - e * e);               // 半通径
    double E = computeEccentricAnomaly(M, e);   // 偏近点角
    double nu = computeTrueAnomaly(E, e);       // 真近点角
    double r = a * (1.0 - e * std::cos(E));     // 地心距
    
    // J2摄动系数
    double factor = (3.0 * J2 * MU * RE * RE) / (2.0 * std::pow(p, 2));
    
    // 计算轨道要素变化率 (da/dt, de/dt, di/dt, dO/dt, dw/dt, dM/dt)
    Eigen::VectorXd derivatives(6);
    
    // 半长轴变化率 - J2摄动不影响半长轴的长期变化
    derivatives[0] = 0.0;
    
    // 偏心率变化率 - J2摄动不影响偏心率的长期变化
    derivatives[1] = 0.0;
    
    // 倾角变化率 - J2摄动不影响倾角的长期变化
    derivatives[2] = 0.0;
    
    // 升交点赤经变化率
    derivatives[3] = -factor * std::cos(i) / (n * a * a);
    
    // 近地点幅角变化率
    derivatives[4] = factor * (2.0 - 2.5 * std::pow(std::sin(i), 2)) / (n * a * a);
    
    // 平近点角变化率
    derivatives[5] = n + factor * ((3.0 * std::pow(std::cos(i), 2) - 1.0) / 2.0) * (1.0 - e * e) / (n * a * a);
    
    return derivatives;
}

OrbitalElements J2OrbitPropagator::rk4Integrate(const OrbitalElements& elements, double dt) {
    // 计算四阶龙格-库塔系数
    Eigen::VectorXd k1 = computeDerivatives(elements);
    
    OrbitalElements temp = elements;
    temp.a += k1[0] * dt / 2.0;
    temp.e += k1[1] * dt / 2.0;
    temp.i += k1[2] * dt / 2.0;
    temp.O += k1[3] * dt / 2.0;
    temp.w += k1[4] * dt / 2.0;
    temp.M += k1[5] * dt / 2.0;
    Eigen::VectorXd k2 = computeDerivatives(temp);
    
    temp = elements;
    temp.a += k2[0] * dt / 2.0;
    temp.e += k2[1] * dt / 2.0;
    temp.i += k2[2] * dt / 2.0;
    temp.O += k2[3] * dt / 2.0;
    temp.w += k2[4] * dt / 2.0;
    temp.M += k2[5] * dt / 2.0;
    Eigen::VectorXd k3 = computeDerivatives(temp);
    
    temp = elements;
    temp.a += k3[0] * dt;
    temp.e += k3[1] * dt;
    temp.i += k3[2] * dt;
    temp.O += k3[3] * dt;
    temp.w += k3[4] * dt;
    temp.M += k3[5] * dt;
    Eigen::VectorXd k4 = computeDerivatives(temp);
    
    // 计算最终的轨道要素
    OrbitalElements result = elements;
    result.a += (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) * dt / 6.0;
    result.e += (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) * dt / 6.0;
    result.i += (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) * dt / 6.0;
    result.O += (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) * dt / 6.0;
    result.w += (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]) * dt / 6.0;
    result.M += (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5]) * dt / 6.0;
    
    // 归一化角度
    result.i = normalizeAngle(result.i);
    result.O = normalizeAngle(result.O);
    result.w = normalizeAngle(result.w);
    result.M = normalizeAngle(result.M);
    
    return result;
}

double J2OrbitPropagator::computeEccentricAnomaly(double M, double e) {
    // 确保平近点角在正确范围内
    M = normalizeAngle(M);
    
    // 初始猜测
    double E;
    if (e < 0.8) {
        E = M;
    } else {
        E = M > M_PI ? M - e : M + e;
    }
    
    // 牛顿迭代法求解开普勒方程
    double delta = 1.0;
    int max_iter = 100;
    int iter = 0;
    
    while (std::abs(delta) > EPSILON && iter < max_iter) {
        delta = (E - e * std::sin(E) - M) / (1.0 - e * std::cos(E));
        E -= delta;
        iter++;
    }
    
    if (iter >= max_iter) {
        std::cerr << "Warning: Eccentric anomaly calculation did not converge." << std::endl;
    }
    
    return E;
}

double J2OrbitPropagator::computeTrueAnomaly(double E, double e) {
    double tan_nu_2 = std::sqrt((1.0 + e) / (1.0 - e)) * std::tan(E / 2.0);
    double nu = 2.0 * std::atan(tan_nu_2);
    return normalizeAngle(nu);
}

double J2OrbitPropagator::normalizeAngle(double angle) {
    angle = std::fmod(angle, 2.0 * M_PI);
    if (angle < 0) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

StateVector J2OrbitPropagator::elementsToState(const OrbitalElements& elements) {
    StateVector state;
    
    double a = elements.a;
    double e = elements.e;
    double i = elements.i;
    double O = elements.O;
    double w = elements.w;
    double M = elements.M;
    
    // 计算偏近点角和真近点角
    double E = computeEccentricAnomaly(M, e);
    double nu = computeTrueAnomaly(E, e);
    
    // 计算地心距
    double r = a * (1.0 - e * std::cos(E));
    
    // 轨道平面内的位置矢量
    double x_perifocal = r * std::cos(nu);
    double y_perifocal = r * std::sin(nu);
    double z_perifocal = 0.0;
    
    // 计算转换矩阵 (惯性系 -> 轨道系)
    double cosO = std::cos(O);
    double sinO = std::sin(O);
    double cosi = std::cos(i);
    double sini = std::sin(i);
    double cosw = std::cos(w);
    double sinw = std::sin(w);
    
    // 旋转矩阵: 升交点赤经 -> 倾角 -> 近地点幅角
    Eigen::Matrix3d R;
    R << cosO*cosw - sinO*sinw*cosi, -cosO*sinw - sinO*cosw*cosi, sinO*sini,
         sinO*cosw + cosO*sinw*cosi, -sinO*sinw + cosO*cosw*cosi, -cosO*sini,
         sinw*sini, cosw*sini, cosi;
    
    // 转换到惯性系
    Eigen::Vector3d r_perifocal(x_perifocal, y_perifocal, z_perifocal);
    state.r = R * r_perifocal;
    
    // 计算速度矢量
    double n = std::sqrt(MU / std::pow(a, 3));
    double x_dot_perifocal = -n * a / std::sqrt(1.0 - e*e) * std::sin(E);
    double y_dot_perifocal = n * a / std::sqrt(1.0 - e*e) * std::sqrt(1.0 - e*e) * std::cos(E);
    double z_dot_perifocal = 0.0;
    
    Eigen::Vector3d v_perifocal(x_dot_perifocal, y_dot_perifocal, z_dot_perifocal);
    state.v = R * v_perifocal;
    
    return state;
}

OrbitalElements J2OrbitPropagator::stateToElements(const StateVector& state, double t) {
    OrbitalElements elements;
    elements.t = t;
    
    Eigen::Vector3d r = state.r;
    Eigen::Vector3d v = state.v;
    
    // 计算角动量矢量
    Eigen::Vector3d h = r.cross(v);
    double h_mag = h.norm();
    
    // 计算偏心率矢量
    Eigen::Vector3d e_vec = (v.cross(h) / MU) - (r / r.norm());
    elements.e = e_vec.norm();
    
    // 计算半长轴
    double r_mag = r.norm();
    double v_mag = v.norm();
    elements.a = 1.0 / (2.0 / r_mag - v_mag * v_mag / MU);
    
    // 计算倾角
    elements.i = std::acos(h.z() / h_mag);
    
    // 计算升交点矢量
    Eigen::Vector3d K(0.0, 0.0, 1.0);
    Eigen::Vector3d N = K.cross(h);
    double N_mag = N.norm();
    
    // 计算升交点赤经
    if (N_mag > EPSILON) {
        elements.O = std::acos(N.x() / N_mag);
        if (N.y() < 0.0) {
            elements.O = 2.0 * M_PI - elements.O;
        }
    } else {
        elements.O = 0.0;  // 赤道轨道，升交点无定义
    }
    
    // 计算近地点幅角
    if (N_mag > EPSILON && elements.e > EPSILON) {
        elements.w = std::acos(N.dot(e_vec) / (N_mag * elements.e));
        if (e_vec.z() < 0.0) {
            elements.w = 2.0 * M_PI - elements.w;
        }
    } else {
        elements.w = 0.0;  // 圆形轨道，近地点无定义
    }
    
    // 计算真近点角
    double nu;
    if (elements.e > EPSILON) {
        nu = std::acos(e_vec.dot(r) / (elements.e * r_mag));
        if (r.dot(v) < 0.0) {
            nu = 2.0 * M_PI - nu;
        }
    } else {
        nu = 0.0;  // 圆形轨道，真近点角无定义
    }
    
    // 计算偏近点角
    double E = 2.0 * std::atan(std::tan(nu / 2.0) * std::sqrt((1.0 - elements.e) / (1.0 + elements.e)));
    
    // 计算平近点角
    elements.M = E - elements.e * std::sin(E);
    elements.M = normalizeAngle(elements.M);
    
    // 归一化所有角度
    elements.i = normalizeAngle(elements.i);
    elements.O = normalizeAngle(elements.O);
    elements.w = normalizeAngle(elements.w);
    
    return elements;
}
