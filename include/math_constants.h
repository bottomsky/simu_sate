#ifndef MATH_CONSTANTS_H
#define MATH_CONSTANTS_H

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

#endif // MATH_CONSTANTS_H