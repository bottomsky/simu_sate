#ifndef MATH_DEFS_H
#define MATH_DEFS_H

#include <cmath>
#include <Eigen/Dense>

// =============================================================================
// 数学常数定义
// =============================================================================

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288  // π
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923132169163975144  // π/2
#endif

#ifndef M_PI_4
#define M_PI_4 0.785398163397448309615660845819875721  // π/4
#endif

#ifndef M_2PI
#define M_2PI 6.28318530717958647692528676655900576  // 2π
#endif

#ifndef M_E
#define M_E 2.71828182845904523536028747135266250  // e
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880168872420969808  // √2
#endif

#ifndef M_SQRT3
#define M_SQRT3 1.73205080756887729352744634150587237  // √3
#endif

#ifndef M_1_PI
#define M_1_PI 0.318309886183790671537767526745028724  // 1/π
#endif

#ifndef M_2_PI
#define M_2_PI 0.636619772367581343075535053490057448  // 2/π
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.707106781186547524400844362104849039  // 1/√2
#endif

// =============================================================================
// 地球物理常数定义
// =============================================================================

// 基本物理常数
const double MU_EARTH = 3.986004418e14;          // 地球标准引力参数 GM (m³/s²)
const double RE_EARTH = 6378137.0;               // 地球赤道半径 WGS84 (m)
const double RF_EARTH = 298.257223563;           // 地球扁率倒数 WGS84
const double OMEGA_EARTH = 7.2921159e-5;         // 地球自转角速度 (rad/s)

// J2000.0参考框架常数
const double J2_EARTH = 1.08263e-3;              // 地球二阶带谐系数 J₂
const double J3_EARTH = -2.532e-6;               // 地球三阶带谐系数 J₃
const double J4_EARTH = -1.619e-6;               // 地球四阶带谐系数 J₄

// 大气模型常数
const double R_GAS = 287.0;                      // 气体常数 (J/kg·K)
const double GAMMA_AIR = 1.4;                    // 空气比热比
const double RHO_0 = 1.225;                      // 海平面大气密度 (kg/m³)
const double H_SCALE = 8500.0;                   // 大气标度高度 (m)

// 太阳辐射压常数
const double C_LIGHT = 299792458.0;              // 光速 (m/s)
const double P_SUN = 4.56e-6;                    // 太阳辐射压 1AU处 (N/m²)
const double AU = 149597870700.0;                // 天文单位 (m)

// =============================================================================
// 数值计算常数定义
// =============================================================================

// 数值精度
const double EPSILON = 1e-12;                    // 默认数值计算精度
const double EPSILON_SMALL = 1e-15;              // 高精度数值计算精度
const double EPSILON_LARGE = 1e-9;               // 低精度数值计算精度

// 角度转换
const double DEG_TO_RAD = M_PI / 180.0;          // 角度转弧度
const double RAD_TO_DEG = 180.0 / M_PI;          // 弧度转角度
const double ARCSEC_TO_RAD = M_PI / (180.0 * 3600.0);  // 角秒转弧度

// 时间转换
const double SEC_PER_DAY = 86400.0;              // 每天秒数
const double SEC_PER_HOUR = 3600.0;              // 每小时秒数
const double SEC_PER_MIN = 60.0;                 // 每分钟秒数
const double DAY_PER_YEAR = 365.25;              // 每年天数(儒略年)
const double SEC_PER_YEAR = DAY_PER_YEAR * SEC_PER_DAY;  // 每年秒数

// J2000.0 历元常数（用于GMST计算）
const double J2000_JD = 2451545.0;               // J2000.0 儒略日
const double GMST_0_J2000 = 18.697374558;        // J2000.0 0时UT1对应的GMST (小时)
const double GMST_RATE = 1.00273790935;          // 平恒星时相对于世界时的速率

// 单位转换
const double KM_TO_M = 1000.0;                   // 千米转米
const double M_TO_KM = 1.0 / 1000.0;             // 米转千米

// =============================================================================
// 兼容性宏定义 (向后兼容)
// =============================================================================

// 为向后兼容保留的别名
#define MU MU_EARTH
#define RE RE_EARTH  
#define J2 J2_EARTH

// =============================================================================
// 内联函数定义
// =============================================================================

// 角度归一化函数
inline double normalizeAngle(double angle) {
    angle = std::fmod(angle, M_2PI);
    if (angle < 0) angle += M_2PI;
    return angle;
}

// 角度归一化到[-π, π]
inline double normalizeAnglePiToPi(double angle) {
    angle = std::fmod(angle + M_PI, M_2PI);
    if (angle < 0) angle += M_2PI;
    return angle - M_PI;
}

// 度分秒转弧度
inline double dmsToRad(int degrees, int minutes, double seconds) {
    return (degrees + minutes / 60.0 + seconds / 3600.0) * DEG_TO_RAD;
}

// 弧度转度分秒
inline void radToDms(double radians, int& degrees, int& minutes, double& seconds) {
    double deg = radians * RAD_TO_DEG;
    degrees = static_cast<int>(deg);
    double min = (deg - degrees) * 60.0;
    minutes = static_cast<int>(min);
    seconds = (min - minutes) * 60.0;
}

// =============================================================================
// ECI/ECEF 坐标转换函数
// =============================================================================

// 计算格林威治平恒星时（GMST）
// 输入：世界时UT1的儒略日JD
// 输出：GMST角度（弧度）
inline double computeGMST(double jd) {
    // 自J2000.0以来的世纪数
    double T = (jd - J2000_JD) / 36525.0;
    
    // GMST计算（IAU-76/FK5标准）
    double gmst_deg = 280.46061837 + 360.98564736629 * (jd - J2000_JD) 
                    + 0.000387933 * T * T - T * T * T / 38710000.0;
                    
    // 归一化到[0, 360)度范围
    gmst_deg = std::fmod(gmst_deg, 360.0);
    if (gmst_deg < 0) gmst_deg += 360.0;
    
    return gmst_deg * DEG_TO_RAD;  // 转换为弧度
}

// 从世界协调时UTC的秒数计算简化的儒略日
// 输入：自某个历元（如Unix时间戳）以来的秒数
// 输出：儒略日
inline double utcSecondsToJulianDay(double utc_seconds) {
    // 这里假设utc_seconds是自J2000.0以来的秒数
    // 在实际应用中可能需要更复杂的UTC到UT1转换
    return J2000_JD + utc_seconds / SEC_PER_DAY;
}

// ECI转ECEF旋转矩阵
// 输入：格林威治平恒星时角度（弧度）
// 输出：3x3旋转矩阵
inline Eigen::Matrix3d eciToEcefRotationMatrix(double gmst) {
    double cos_gmst = std::cos(gmst);
    double sin_gmst = std::sin(gmst);
    
    Eigen::Matrix3d R;
    R << cos_gmst,  sin_gmst, 0,
        -sin_gmst,  cos_gmst, 0,
               0,        0, 1;
    
    return R;
}

// ECEF转ECI旋转矩阵（ECI转ECEF的转置）
// 输入：格林威治平恒星时角度（弧度）
// 输出：3x3旋转矩阵
inline Eigen::Matrix3d ecefToEciRotationMatrix(double gmst) {
    return eciToEcefRotationMatrix(gmst).transpose();
}

// ECI位置矢量转ECEF位置矢量
// 输入：ECI位置矢量（m），时间（自J2000.0以来的秒数）
// 输出：ECEF位置矢量（m）
inline Eigen::Vector3d eciToEcefPosition(const Eigen::Vector3d& r_eci, double time_j2000_seconds) {
    double jd = utcSecondsToJulianDay(time_j2000_seconds);
    double gmst = computeGMST(jd);
    Eigen::Matrix3d R = eciToEcefRotationMatrix(gmst);
    return R * r_eci;
}

// ECEF位置矢量转ECI位置矢量
// 输入：ECEF位置矢量（m），时间（自J2000.0以来的秒数）
// 输出：ECI位置矢量（m）
inline Eigen::Vector3d ecefToEciPosition(const Eigen::Vector3d& r_ecef, double time_j2000_seconds) {
    double jd = utcSecondsToJulianDay(time_j2000_seconds);
    double gmst = computeGMST(jd);
    Eigen::Matrix3d R = ecefToEciRotationMatrix(gmst);
    return R * r_ecef;
}

// ECI速度矢量转ECEF速度矢量
// 输入：ECI位置矢量（m），ECI速度矢量（m/s），时间（自J2000.0以来的秒数）
// 输出：ECEF速度矢量（m/s）
inline Eigen::Vector3d eciToEcefVelocity(const Eigen::Vector3d& r_eci, const Eigen::Vector3d& v_eci, 
                                         double time_j2000_seconds) {
    double jd = utcSecondsToJulianDay(time_j2000_seconds);
    double gmst = computeGMST(jd);
    Eigen::Matrix3d R = eciToEcefRotationMatrix(gmst);
    
    // 地球自转矢量（沿Z轴）
    Eigen::Vector3d omega_earth(0, 0, OMEGA_EARTH);
    
    // 速度转换：v_ecef = R * v_eci - omega_earth × r_ecef
    Eigen::Vector3d r_ecef = R * r_eci;
    return R * v_eci - omega_earth.cross(r_ecef);
}

// ECEF速度矢量转ECI速度矢量
// 输入：ECEF位置矢量（m），ECEF速度矢量（m/s），时间（自J2000.0以来的秒数）
// 输出：ECI速度矢量（m/s）
inline Eigen::Vector3d ecefToEciVelocity(const Eigen::Vector3d& r_ecef, const Eigen::Vector3d& v_ecef,
                                         double time_j2000_seconds) {
    double jd = utcSecondsToJulianDay(time_j2000_seconds);
    double gmst = computeGMST(jd);
    Eigen::Matrix3d R = ecefToEciRotationMatrix(gmst);
    
    // 地球自转矢量（沿Z轴）
    Eigen::Vector3d omega_earth(0, 0, OMEGA_EARTH);
    
    // 速度转换：v_eci = R * (v_ecef + omega_earth × r_ecef)
    return R * (v_ecef + omega_earth.cross(r_ecef));
}

#endif // MATH_DEFS_H