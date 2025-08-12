/**
 * @file j2_orbit_propagator.cpp
 * @brief J2摄动轨道传播器的实现文件。
 *
 * 该文件实现了 J2OrbitPropagator 类的成员函数，该类用于模拟和预测卫星在考虑地球J2摄动影响下的轨道运动。
 * 主要功能包括：
 * - 使用四阶龙格-库塔（RK4）积分器进行轨道外推。
 * - 支持固定步长和自适应步长两种积分模式。
 * - 提供了轨道根数与状态向量（位置和速度）之间的转换方法。
 * - 实现了求解开普勒方程以计算偏近点角的数值方法。
 */

#include "j2_orbit_propagator.h"

/**
 * @brief J2OrbitPropagator 类的构造函数。
 * @param initial_elements 卫星的初始轨道根数。
 */
J2OrbitPropagator::J2OrbitPropagator(const OrbitalElements& initial_elements) 
    : current_elements_(initial_elements), 
      step_size_(60.0), // 默认步长设为60秒
      adaptive_step_size_(false), 
      tolerance_(1e-6), 
      min_step_size_(1.0), 
      max_step_size_(300.0) 
{
    // 构造时立即对输入的角度进行归一化，确保其在 [0, 2*PI) 范围内。
    current_elements_.i = normalizeAngle(current_elements_.i);
    current_elements_.O = normalizeAngle(current_elements_.O);
    current_elements_.w = normalizeAngle(current_elements_.w);
    current_elements_.M = normalizeAngle(current_elements_.M);
}

/**
 * @brief 将卫星轨道外推到指定的时间点。
 * @param t 目标时间点（秒）。
 * @return 在目标时间点的轨道根数。
 */
OrbitalElements J2OrbitPropagator::propagate(double t) {
    // 计算总的外推时间。
    double dt_total = t - current_elements_.t;
    
    // 不支持时间倒推。
    if (dt_total < 0) {
        std::cerr << "Error: Target time is earlier than epoch time." << std::endl;
        return current_elements_;
    }
    
    // 如果时间差非常小，则无需外推，直接返回当前状态。
    if (dt_total < EPSILON) {
        return current_elements_;
    }
    
    double remaining_time = dt_total;

    // 根据是否启用自适应步长，选择不同的积分循环。
    if (!adaptive_step_size_) {
        // --- 固定步长积分 ---
        // 使用固定的步长 `step_size_` 进行迭代，直到到达目标时间。
        while (remaining_time > EPSILON) {
            double dt = std::min(remaining_time, step_size_);
            current_elements_ = rk4Integrate(current_elements_, dt);
            current_elements_.t += dt;
            remaining_time -= dt;
        }
    } else {
        // --- 自适应步长积分 ---
        // 初始步长从用户设定的默认步长和最大/最小步长之间选择。
        double dt = std::min(remaining_time, step_size_);
        dt = std::max(min_step_size_, std::min(dt, max_step_size_));

        while (remaining_time > EPSILON) {
            // 确保最后一步恰好到达目标时间。
            if (dt > remaining_time) {
                dt = remaining_time;
            }

            // 估计当前步长的局部截断误差。
            double err = estimateLocalError(current_elements_, dt);

            // 判断误差是否在容差范围内，或者步长是否已达到下限。
            if (err <= tolerance_ || dt <= min_step_size_ + 1e-12) {
                // 步长可接受：更新轨道状态。
                current_elements_ = rk4Integrate(current_elements_, dt);
                current_elements_.t += dt;
                remaining_time -= dt;

                // 成功后，尝试增大学下一步的步长以提高效率。
                // 步长调整策略基于经典的PI控制器思想。
                // safety: 安全因子，防止步长增长过快。
                // growth: 步长增长的上限因子。
                // pow(err, -0.2): 误差越小，步长增加得越多。指数-0.2是RK4/5方法的理论最优值。
                if (err > 1e-16) { // 避免除以零
                    double safety = 0.9;
                    double growth_factor = 1.5;
                    double error_ratio = err / tolerance_;
                    double new_dt = dt * safety * std::pow(error_ratio, -0.2);
                    dt = std::min(new_dt, dt * growth_factor); // 限制单次增长幅度
                } else {
                    dt *= 1.5; // 如果误差极小，直接按比例增加
                }

            } else {
                // 步长不可接受：减小步长后重试当前时间步。
                // safety: 安全因子，确保新步长足够小。
                // pow(err, -0.25): 误差越大，步长减小得越多。指数-0.25是RK4方法的保守选择。
                double safety = 0.9;
                double error_ratio = err / tolerance_;
                dt = dt * safety * std::pow(error_ratio, -0.25);
            }
            // 确保新计算的步长在设定的最大和最小步长范围内。
            dt = std::max(min_step_size_, std::min(dt, max_step_size_));
        }
    }
    
    // 最终将历元时间精确设置为目标时间。
    current_elements_.t = t;
    
    return current_elements_;
}

/**
 * @brief 计算J2摄动下轨道根数对时间的变化率。
 * @param elements 当前的轨道根数。
 * @return 一个包含6个轨道根数变化率的Eigen向量 (da/dt, de/dt, di/dt, dO/dt, dw/dt, dM/dt)。
 */
Eigen::VectorXd J2OrbitPropagator::computeDerivatives(const OrbitalElements& elements) {
    double a = elements.a;
    double e = elements.e;
    double i = elements.i;
    
    // 在J2摄动的长期平均模型中，半长轴(a)、偏心率(e)和倾角(i)不发生长期变化。
    // 因此它们的导数被设为0。
    Eigen::VectorXd derivatives = Eigen::VectorXd::Zero(6);
    
    // 避免奇异性：当偏心率接近1时（抛物线轨道），分母会为零。
    if (std::abs(1.0 - e * e) < EPSILON) {
        return derivatives;
    }

    // 计算公共计算因子，以提高效率。
    double n = std::sqrt(MU / (a * a * a)); // 平均角速度
    double p = a * (1.0 - e * e);           // 半通径
    double factor = (3.0 / 2.0) * J2 * n * (RE / p) * (RE / p);

    double cos_i = std::cos(i);
    double sin_i_sq = std::sin(i) * std::sin(i);

    // 升交点赤经(O)的变化率 (dO/dt)
    derivatives[3] = -factor * cos_i;
    
    // 近地点幅角(w)的变化率 (dw/dt)
    derivatives[4] = factor * (2.0 - 2.5 * sin_i_sq);
    
    // 平近点角(M)的变化率 (dM/dt)
    // 包含自然运动(n)和J2摄动引起的附加项。
    // 注意：J2摄动项应为负值，因为J2效应会减缓卫星的平均运动
    derivatives[5] = n - factor * std::sqrt(1.0 - e * e) * (1.5 * sin_i_sq - 0.5);
    
    return derivatives;
}

/**
 * @brief 使用经典的四阶龙格-库塔（RK4）方法执行单步积分。
 * @param elements 积分开始时的轨道根数。
 * @param dt 积分步长（秒）。
 * @return 积分结束时的轨道根数。
 */
OrbitalElements J2OrbitPropagator::rk4Integrate(const OrbitalElements& elements, double dt) {
    // RK4方法通过在当前步内计算4个不同点的导数并加权平均来提高精度。
    
    // k1: 在步长起点计算导数
    Eigen::VectorXd k1 = computeDerivatives(elements);
    
    // k2: 在步长中点使用k1预测的值计算导数
    OrbitalElements temp = elements;
    temp.O = normalizeAngle(temp.O + k1[3] * dt / 2.0);
    temp.w = normalizeAngle(temp.w + k1[4] * dt / 2.0);
    temp.M = normalizeAngle(temp.M + k1[5] * dt / 2.0);
    Eigen::VectorXd k2 = computeDerivatives(temp);
    
    // k3: 在步长中点使用k2预测的值计算导数
    temp = elements;
    temp.O = normalizeAngle(temp.O + k2[3] * dt / 2.0);
    temp.w = normalizeAngle(temp.w + k2[4] * dt / 2.0);
    temp.M = normalizeAngle(temp.M + k2[5] * dt / 2.0);
    Eigen::VectorXd k3 = computeDerivatives(temp);
    
    // k4: 在步长终点使用k3预测的值计算导数
    temp = elements;
    temp.O = normalizeAngle(temp.O + k3[3] * dt);
    temp.w = normalizeAngle(temp.w + k3[4] * dt);
    temp.M = normalizeAngle(temp.M + k3[5] * dt);
    Eigen::VectorXd k4 = computeDerivatives(temp);
    
    // 将k1, k2, k3, k4加权平均，更新轨道根数。
    OrbitalElements result = elements;
    result.O += (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) * dt / 6.0;
    result.w += (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]) * dt / 6.0;
    result.M += (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5]) * dt / 6.0;
    
    // 积分后对所有角度进行归一化。
    result.i = normalizeAngle(result.i);
    result.O = normalizeAngle(result.O);
    result.w = normalizeAngle(result.w);
    result.M = normalizeAngle(result.M);
    
    return result;
}

/**
 * @brief 估计单步积分的局部截断误差。
 *
 * 该方法通过“步长加倍”技术实现：比较一次大步长（dt）积分的结果与两次小步长（dt/2）积分的结果。
 * 两者之差可以作为局部截断误差的估计值。
 *
 * @param elements 当前轨道根数。
 * @param dt 当前积分步长。
 * @return 一个无量纲的组合误差值。
 */
double J2OrbitPropagator::estimateLocalError(const OrbitalElements& elements, double dt) {
    // y1: 使用完整步长 dt 进行一次积分。
    OrbitalElements y1 = rk4Integrate(elements, dt);
    
    // y2: 使用一半的步长 dt/2 进行两次连续积分。
    OrbitalElements half_step_result = rk4Integrate(elements, dt * 0.5);
    OrbitalElements y2 = rk4Integrate(half_step_result, dt * 0.5);
    
    // 定义一个辅助lambda函数来计算两个角度之间的最小差值。
    auto angle_diff = [&](double a1, double a2){
        double d = std::fmod(std::abs(a1 - a2), 2.0 * M_PI);
        return std::min(d, 2.0 * M_PI - d);
    };

    // 计算每个轨道根数的误差。
    // 对于半长轴和偏心率，使用相对误差。
    // 对于角度，使用绝对差值。
    double err_a = std::abs(y1.a - y2.a) / std::max(1.0, std::abs(elements.a));
    double err_e = std::abs(y1.e - y2.e) / std::max(1e-12, std::abs(elements.e));
    double err_i = angle_diff(y1.i, y2.i);
    double err_O = angle_diff(y1.O, y2.O);
    double err_w = angle_diff(y1.w, y2.w);
    double err_M = angle_diff(y1.M, y2.M);
    
    // 将所有误差组合成一个单一的标量值。
    // 权重可以根据不同根数的重要性进行调整。
    // 角度误差通过除以 2*PI 进行归一化。
    double total_error = err_a + err_e + (err_i + err_O + err_w + err_M) / (2.0 * M_PI);
    
    // 引入偏心率缩放因子：对于高偏心率轨道，误差容忍度可以适当放宽，
    // 因为在近地点附近动力学变化剧烈，很难维持小步长。
    double eccentricity_scale = 1.0 / std::sqrt(1.0 - elements.e * elements.e);
    
    return total_error / eccentricity_scale;
}

/**
 * @brief 使用牛顿迭代法求解开普勒方程 M = E - e*sin(E)，以计算偏近点角 E。
 * @param M 平近点角（弧度）。
 * @param e 偏心率。
 * @return 偏近点角 E（弧度）。
 */
double J2OrbitPropagator::computeEccentricAnomaly(double M, double e) {
    // 将平近点角归一化到 [0, 2*PI) 范围。
    M = normalizeAngle(M);
    
    // 根据偏心率选择一个合适的初始猜测值 E0，这有助于加速收敛。
    double E = (e < 0.8) ? M : M_PI;
    
    // 牛顿法迭代求解。
    double f_E, df_E;
    int max_iter = 100;
    for (int i = 0; i < max_iter; ++i) {
        f_E = E - e * std::sin(E) - M;
        if (std::abs(f_E) < EPSILON) {
            break; // 当误差足够小时，停止迭代。
        }
        df_E = 1.0 - e * std::cos(E);
        E = E - f_E / df_E;
    }
    
    return E;
}

/**
 * @brief 根据偏近点角 E 和偏心率 e 计算真近点角 nu。
 * @param E 偏近点角（弧度）。
 * @param e 偏心率。
 * @return 真近点角 nu（弧度）。
 */
double J2OrbitPropagator::computeTrueAnomaly(double E, double e) {
    // 使用半角公式计算，可以提高数值稳定性。
    double tan_nu_2 = std::sqrt((1.0 + e) / (1.0 - e)) * std::tan(E / 2.0);
    double nu = 2.0 * std::atan(tan_nu_2);
    return normalizeAngle(nu);
}

/**
 * @brief 将角度归一化到 [0, 2*PI) 范围内。
 * @param angle 要归一化的角度（弧度）。
 * @return 归一化后的角度（弧度）。
 */
double J2OrbitPropagator::normalizeAngle(double angle) {
    angle = std::fmod(angle, 2.0 * M_PI);
    if (angle < 0) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

/**
 * @brief 将轨道根数转换为地心惯性系（ECI）下的状态向量（位置和速度）。
 * @param elements 轨道根数。
 * @return 包含位置和速度的状态向量。
 */
StateVector J2OrbitPropagator::elementsToState(const OrbitalElements& elements) {
    StateVector state;
    
    double a = elements.a;
    double e = elements.e;
    double i = elements.i;
    double O = elements.O;
    double w = elements.w;
    double M = elements.M;
    
    // 1. 从平近点角 M 计算偏近点角 E。
    double E = computeEccentricAnomaly(M, e);
    
    // 2. 从偏近点角 E 计算真近点角 nu。
    double nu = computeTrueAnomaly(E, e);
    
    // 3. 计算地心距 r。
    double r_mag = a * (1.0 - e * std::cos(E));
    
    // 4. 在轨道平面（周航坐标系）内计算位置和速度。
    Eigen::Vector3d r_perifocal(r_mag * std::cos(nu), r_mag * std::sin(nu), 0.0);
    
    double p = a * (1.0 - e * e); // 半通径
    double v_mag_factor = std::sqrt(MU / p);
    Eigen::Vector3d v_perifocal(-v_mag_factor * std::sin(nu), v_mag_factor * (e + std::cos(nu)), 0.0);

    // 5. 构建从周航坐标系到地心惯性系（ECI）的旋转矩阵。
    // 这是一个 3-1-3 欧拉角旋转：Rz(O) * Rx(i) * Rz(w) 的转置。
    Eigen::Matrix3d R;
    double cosO = std::cos(O), sinO = std::sin(O);
    double cosi = std::cos(i), sini = std::sin(i);
    double cosw = std::cos(w), sinw = std::sin(w);
    
    R(0,0) = cosO * cosw - sinO * sinw * cosi;
    R(0,1) = -cosO * sinw - sinO * cosw * cosi;
    R(0,2) = sinO * sini;
    
    R(1,0) = sinO * cosw + cosO * sinw * cosi;
    R(1,1) = -sinO * sinw + cosO * cosw * cosi;
    R(1,2) = -cosO * sini;
    
    R(2,0) = sinw * sini;
    R(2,1) = cosw * sini;
    R(2,2) = cosi;
    
    // 6. 将周航坐标系下的位置和速度旋转到ECI系。
    state.r = R * r_perifocal;
    state.v = R * v_perifocal;
    
    return state;
}

/**
 * @brief 将地心惯性系（ECI）下的状态向量（位置和速度）转换为轨道根数。
 * @param state 包含位置和速度的状态向量。
 * @param t 对应的历元时间。
 * @return 轨道根数。
 */
OrbitalElements J2OrbitPropagator::stateToElements(const StateVector& state, double t) {
    OrbitalElements elements;
    elements.t = t;
    
    const Eigen::Vector3d& r_vec = state.r;
    const Eigen::Vector3d& v_vec = state.v;
    double r = r_vec.norm();
    double v = v_vec.norm();
    
    // 1. 计算角动量矢量 h。
    Eigen::Vector3d h_vec = r_vec.cross(v_vec);
    double h = h_vec.norm();
    
    // 2. 计算升交点矢量 n。
    Eigen::Vector3d K_hat(0, 0, 1);
    Eigen::Vector3d n_vec = K_hat.cross(h_vec);
    double n = n_vec.norm();
    
    // 3. 计算偏心率矢量 e。
    Eigen::Vector3d e_vec = ((v*v - MU/r) * r_vec - (r_vec.dot(v_vec)) * v_vec) / MU;
    elements.e = e_vec.norm();
    
    // 4. 计算轨道能量 Epsilon，并从中得到半长轴 a。
    double energy = v*v/2.0 - MU/r;
    elements.a = -MU / (2.0 * energy);
    
    // 5. 计算倾角 i。
    elements.i = std::acos(h_vec.z() / h);
    
    // 6. 计算升交点赤经 O。
    elements.O = std::acos(n_vec.x() / n);
    if (n_vec.y() < 0) {
        elements.O = 2.0 * M_PI - elements.O;
    }
    
    // 7. 计算近地点幅角 w。
    elements.w = std::acos(n_vec.dot(e_vec) / (n * elements.e));
    if (e_vec.z() < 0) {
        elements.w = 2.0 * M_PI - elements.w;
    }
    
    // 8. 计算真近点角 nu。
    double nu = std::acos(e_vec.dot(r_vec) / (elements.e * r));
    if (r_vec.dot(v_vec) < 0) {
        nu = 2.0 * M_PI - nu;
    }
    
    // 9. 从真近点角 nu 计算偏近点角 E。
    // 使用更稳定的公式并处理象限问题
    double E;
    if (std::abs(elements.e) < EPSILON) {
        // 圆轨道情况
        E = nu;
    } else {
        // 椭圆轨道情况
        double sin_E = std::sqrt(1.0 - elements.e * elements.e) * std::sin(nu) / (1.0 + elements.e * std::cos(nu));
        double cos_E = (elements.e + std::cos(nu)) / (1.0 + elements.e * std::cos(nu));
        E = std::atan2(sin_E, cos_E);
        E = normalizeAngle(E);
    }
    
    // 10. 从偏近点角 E 计算平近点角 M。
    elements.M = E - elements.e * std::sin(E);
    
    // 归一化所有角度。
    elements.i = normalizeAngle(elements.i);
    elements.O = normalizeAngle(elements.O);
    elements.w = normalizeAngle(elements.w);
    elements.M = normalizeAngle(elements.M);
    
    return elements;
}
