using System;
using System.IO;
using System.Runtime.InteropServices;

namespace J2.Propagator
{
    // 具体异常类型映射
    /// <summary>
    /// 表示对原生库（P/Invoke）调用失败的基类异常。
    /// </summary>
    public class NativeCallException : Exception 
    { 
        /// <summary>
        /// 初始化 <see cref="NativeCallException"/> 类的新实例。
        /// </summary>
        /// <param name="msg">错误消息。</param>
        public NativeCallException(string msg) : base(msg) {}
    }
    /// <summary>
    /// 表示轨道传播过程失败时抛出的异常。
    /// </summary>
    public sealed class PropagationException : NativeCallException 
    { 
        /// <summary>
        /// 初始化 <see cref="PropagationException"/> 类的新实例。
        /// </summary>
        /// <param name="msg">错误消息。</param>
        public PropagationException(string msg) : base(msg) {}
    }
    /// <summary>
    /// 表示坐标/要素转换过程失败时抛出的异常。
    /// </summary>
    public sealed class ConversionException : NativeCallException 
    { 
        /// <summary>
        /// 初始化 <see cref="ConversionException"/> 类的新实例。
        /// </summary>
        /// <param name="msg">错误消息。</param>
        public ConversionException(string msg) : base(msg) {}
    }

    // 与C结构体内存布局保持一致
    /// <summary>
    /// 轨道根数（与原生 C 结构体二进制布局一致）。
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 8)]
    public struct COrbitalElements
    {
        /// <summary>半长轴 a（米）。</summary>
        public double a;
        /// <summary>偏心率 e（无量纲，0-1）。</summary>
        public double e;
        /// <summary>轨道倾角 i（弧度）。</summary>
        public double i;
        /// <summary>升交点赤经 Ω（弧度）。</summary>
        public double O;
        /// <summary>近地点幅角 ω（弧度）。</summary>
        public double w;
        /// <summary>平近点角 M（弧度）。</summary>
        public double M;
        /// <summary>历元时间 t（秒，参考 J2000.0 = 2000-01-01T12:00:00Z）。</summary>
        public double t;
    }

    /// <summary>
    /// 压紧轨道根数（不包含历元时间，与原生 C 结构体二进制布局一致）。
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 8)]
    public struct CCompactOrbitalElements
    {
        /// <summary>半长轴 a（米）。</summary>
        public double a;
        /// <summary>偏心率 e（无量纲，0-1）。</summary>
        public double e;
        /// <summary>轨道倾角 i（弧度）。</summary>
        public double i;
        /// <summary>升交点赤经 Ω（弧度）。</summary>
        public double O;
        /// <summary>近地点幅角 ω（弧度）。</summary>
        public double w;
        /// <summary>平近点角 M（弧度）。</summary>
        public double M;
    }

    /// <summary>
    /// 位置-速度状态向量（与原生 C 结构体二进制布局一致）。
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 8)]
    public struct CStateVector
    {
        /// <summary>位置向量 r（ECI，米），长度为 3。</summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public double[] r;
        /// <summary>速度向量 v（ECI，米/秒），长度为 3。</summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public double[] v;

        /// <summary>
        /// 创建并初始化包含 3 元素 r、v 数组的 <see cref="CStateVector"/>。
        /// </summary>
        public static CStateVector Create()
        {
            return new CStateVector { r = new double[3], v = new double[3] };
        }
    }

    /// <summary>
    /// 计算模式枚举（与原生 C 枚举保持一致）。
    /// </summary>
    public enum ComputeMode : int
    {
        /// <summary>CPU 标量计算。</summary>
        CpuScalar = 0,
        /// <summary>CPU SIMD 计算。</summary>
        CpuSimd = 1,
        /// <summary>GPU CUDA 计算。</summary>
        GpuCuda = 2
    }

    // 地理经纬高（弧度 + 米），提供度制便捷属性
    /// <summary>
    /// 大地坐标（经纬高），纬度/经度以弧度表示，高程以米表示，并提供度制便捷属性。
    /// </summary>
    public struct GeodeticCoord
    {
        /// <summary>纬度（弧度）。北纬为正，范围约 [-π/2, π/2]。</summary>
        public double LatRad;   // 纬度 (rad)
        /// <summary>经度（弧度）。东经为正，范围 (-π, π]。</summary>
        public double LonRad;   // 经度 (rad)
        /// <summary>大地高（米）。</summary>
        public double AltMeters; // 高程 (m)

        /// <summary>以度表示的纬度（与 <see cref="LatRad"/> 对应）。</summary>
        public double LatDeg { get => LatRad * 180.0 / Math.PI; set => LatRad = value * Math.PI / 180.0; }
        /// <summary>以度表示的经度（与 <see cref="LonRad"/> 对应）。</summary>
        public double LonDeg { get => LonRad * 180.0 / Math.PI; set => LonRad = value * Math.PI / 180.0; }

        /// <summary>
        /// 以度制输入创建大地坐标。
        /// </summary>
        /// <param name="latDeg">纬度（度）。</param>
        /// <param name="lonDeg">经度（度）。</param>
        /// <param name="altMeters">高程（米）。</param>
        public static GeodeticCoord FromDegrees(double latDeg, double lonDeg, double altMeters)
            => new GeodeticCoord { LatRad = latDeg * Math.PI / 180.0, LonRad = lonDeg * Math.PI / 180.0, AltMeters = altMeters };
    }

    // 基本常量（与C++侧保持一致的WGS84）
    internal static class EarthModel
    {
        public const double RE = 6378137.0;              // WGS84 赤道半径 (m)
        public const double RF = 298.257223563;          // 扁率倒数
        public static readonly double f = 1.0 / RF;
        public static readonly double e2 = f * (2.0 - f);          // 第一偏心率平方
        public static readonly double b = RE * (1.0 - f);          // 极半径
        public static readonly double ep2 = (RE * RE - b * b) / (b * b); // 第二偏心率平方
    }

    // UTC时间工具：与C侧约定相同，秒数基于 J2000.0 = 2000-01-01T12:00:00Z
    /// <summary>
    /// UTC 与相对 J2000.0（2000-01-01T12:00:00Z）秒数的相互转换工具。
    /// </summary>
    public static class TimeUtil
    {
        /// <summary>
        /// J2000.0 历元（2000-01-01T12:00:00Z）。
        /// </summary>
        public static readonly DateTime J2000EpochUtc = new DateTime(2000, 1, 1, 12, 0, 0, DateTimeKind.Utc);
        /// <summary>
        /// 将 UTC 时间转换为相对 J2000.0 的秒数。
        /// </summary>
        /// <param name="utc">UTC 时间。</param>
        /// <returns>自 J2000.0 起的秒数（可为负）。</returns>
        public static double ToSecondsSinceJ2000(DateTime utc)
        {
            if (utc.Kind != DateTimeKind.Utc) utc = utc.ToUniversalTime();
            return (utc - J2000EpochUtc).TotalSeconds;
        }
        /// <summary>
        /// 将相对 J2000.0 的秒数转换为 UTC 时间。
        /// </summary>
        /// <param name="seconds">自 J2000.0 起的秒数。</param>
        /// <returns>对应的 UTC 时间。</returns>
        public static DateTime FromSecondsSinceJ2000(double seconds)
            => J2000EpochUtc.AddSeconds(seconds);
    }

    // 地理坐标与ECEF互转
    /// <summary>
    /// 提供大地坐标与 ECEF 坐标之间的相互转换。
    /// </summary>
    public static class GeoConversion
    {
        // ECEF -> 大地坐标（Bowring公式）
        /// <summary>
        /// 将 ECEF 坐标转换为大地坐标（使用 Bowring 公式）。
        /// </summary>
        /// <param name="x">ECEF X（米）。</param>
        /// <param name="y">ECEF Y（米）。</param>
        /// <param name="z">ECEF Z（米）。</param>
        /// <returns>大地坐标（弧度经纬，高程米）。</returns>
        public static GeodeticCoord EcefToGeodetic(double x, double y, double z)
        {
            double a = EarthModel.RE;
            double b = EarthModel.b;
            double e2 = EarthModel.e2;
            double ep2 = EarthModel.ep2;

            double lon = Math.Atan2(y, x);
            double p = Math.Sqrt(x * x + y * y);
            if (p < 1e-12)
            {
                // 极点特殊处理：经度任意，约定为0
                double latPole = Math.Sign(z) * Math.PI / 2.0;
                double Np = a / Math.Sqrt(1.0 - e2 * Math.Sin(latPole) * Math.Sin(latPole));
                double hPole = Math.Abs(z) - Np * (1.0 - e2);
                return new GeodeticCoord { LatRad = latPole, LonRad = 0.0, AltMeters = hPole };
            }

            double theta = Math.Atan2(z * a, p * b);
            double s = Math.Sin(theta);
            double c = Math.Cos(theta);
            double lat = Math.Atan2(z + ep2 * b * s * s * s, p - e2 * a * c * c * c);
            double sinLat = Math.Sin(lat);
            double N = a / Math.Sqrt(1.0 - e2 * sinLat * sinLat);
            double h = p / Math.Cos(lat) - N;

            // 归一化经度到(-pi, pi]
            if (lon <= -Math.PI) lon += 2 * Math.PI;
            if (lon > Math.PI) lon -= 2 * Math.PI;

            return new GeodeticCoord { LatRad = lat, LonRad = lon, AltMeters = h };
        }

        // 大地坐标 -> ECEF
        /// <summary>
        /// 将大地坐标转换为 ECEF 坐标。
        /// </summary>
        /// <param name="geo">大地坐标（弧度经纬，高程米）。</param>
        /// <returns>ECEF 坐标 (x, y, z)（米）。</returns>
        public static (double x, double y, double z) GeodeticToEcef(GeodeticCoord geo)
        {
            double a = EarthModel.RE;
            double e2 = EarthModel.e2;
            double sinLat = Math.Sin(geo.LatRad);
            double cosLat = Math.Cos(geo.LatRad);
            double cosLon = Math.Cos(geo.LonRad);
            double sinLon = Math.Sin(geo.LonRad);
            double N = a / Math.Sqrt(1.0 - e2 * sinLat * sinLat);
            double x = (N + geo.AltMeters) * cosLat * cosLon;
            double y = (N + geo.AltMeters) * cosLat * sinLon;
            double z = (N * (1.0 - e2) + geo.AltMeters) * sinLat;
            return (x, y, z);
        }
    }

    /// <summary>
    /// 跨平台动态库加载器，根据操作系统自动选择正确的动态库文件。
    /// </summary>
    internal static class NativeLibraryLoader
    {
        private static readonly string LibraryName;
        private static readonly string LibraryPath;
        
        static NativeLibraryLoader()
        {
            // 根据操作系统选择正确的库文件名
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                LibraryName = "j2_orbit_propagator.dll";
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                LibraryName = "libj2_orbit_propagator.so";
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                LibraryName = "libj2_orbit_propagator.dylib";
            }
            else
            {
                throw new PlatformNotSupportedException($"Unsupported platform: {RuntimeInformation.OSDescription}");
            }
            
            // 获取库文件的完整路径
            var assemblyDir = Path.GetDirectoryName(typeof(NativeLibraryLoader).Assembly.Location);
            LibraryPath = Path.Combine(assemblyDir ?? "", LibraryName);
            
            // 验证库文件是否存在
            if (!File.Exists(LibraryPath))
            {
                throw new FileNotFoundException($"Native library not found: {LibraryPath}. Please ensure the library is copied to the output directory.");
            }
        }
        
        /// <summary>
        /// 获取当前平台的动态库名称。
        /// </summary>
        public static string GetLibraryName() => LibraryName;
        
        /// <summary>
        /// 获取当前平台的动态库完整路径。
        /// </summary>
        public static string GetLibraryPath() => LibraryPath;
        
        /// <summary>
        /// 加载原生库。
        /// </summary>
        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        private static extern IntPtr LoadLibrary(string lpFileName);
        
        [DllImport("libdl.so.2", SetLastError = true, CharSet = CharSet.Ansi)]
        private static extern IntPtr dlopen(string filename, int flags);
        
        private static IntPtr _libraryHandle = IntPtr.Zero;
        
        /// <summary>
        /// 确保原生库已加载。
        /// </summary>
        public static void EnsureLibraryLoaded()
        {
            if (_libraryHandle != IntPtr.Zero) return;
            
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                _libraryHandle = LoadLibrary(LibraryPath);
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) || RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                const int RTLD_NOW = 2;
                _libraryHandle = dlopen(LibraryPath, RTLD_NOW);
            }
            
            if (_libraryHandle == IntPtr.Zero)
            {
                throw new DllNotFoundException($"Failed to load native library: {LibraryPath}");
            }
        }
    }

    internal static class Native
    {
        private static readonly string LibName = NativeLibraryLoader.GetLibraryName();
        
        static Native()
        {
            // 确保在使用 P/Invoke 之前加载正确的动态库
            NativeLibraryLoader.EnsureLibraryLoaded();
        }
        
        // 原J2传播器函数 (保持原有)
        [DllImport("j2_orbit_propagator", EntryPoint = "j2_propagator_create", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr j2_propagator_create(ref COrbitalElements elements);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_propagator_destroy", CallingConvention = CallingConvention.Cdecl)]
        public static extern void j2_propagator_destroy(IntPtr handle);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_propagator_propagate", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_propagator_propagate(IntPtr handle, double target_time, out COrbitalElements result);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_propagator_elements_to_state", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_propagator_elements_to_state(IntPtr handle, ref COrbitalElements elements, ref CStateVector state);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_propagator_state_to_elements", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_propagator_state_to_elements(IntPtr handle, ref CStateVector state, double time, out COrbitalElements elements);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_propagator_apply_impulse", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_propagator_apply_impulse(IntPtr handle, ref COrbitalElements elements, double[] deltaV, double t, out COrbitalElements result);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_propagator_set_step_size", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_propagator_set_step_size(IntPtr handle, double step_size);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_propagator_get_step_size", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_propagator_get_step_size(IntPtr handle, out double step_size);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_propagator_set_adaptive_step_size", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_propagator_set_adaptive_step_size(IntPtr handle, int enable);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_propagator_set_adaptive_parameters", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_propagator_set_adaptive_parameters(IntPtr handle, double tol, double minStep, double maxStep);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_compute_gmst", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_compute_gmst(double utcSeconds, out double gmst);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_normalize_angle", CallingConvention = CallingConvention.Cdecl)]
        public static extern double j2_normalize_angle(double angle);

        // 位置/速度：ECI <-> ECEF
        [DllImport("j2_orbit_propagator", EntryPoint = "j2_eci_to_ecef_position", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_eci_to_ecef_position([In] double[] eciPosition, double utcSeconds, [Out] double[] ecefPosition);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_ecef_to_eci_position", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_ecef_to_eci_position([In] double[] ecefPosition, double utcSeconds, [Out] double[] eciPosition);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_eci_to_ecef_velocity", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_eci_to_ecef_velocity([In] double[] eciPosition, [In] double[] eciVelocity, double utcSeconds, [Out] double[] ecefVelocity);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_ecef_to_eci_velocity", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_ecef_to_eci_velocity([In] double[] ecefPosition, [In] double[] ecefVelocity, double utcSeconds, [Out] double[] eciVelocity);

        // 地理坐标与 ECEF/ECI 互转
        [DllImport("j2_orbit_propagator", EntryPoint = "j2_ecef_to_geodetic", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_ecef_to_geodetic([In] double[] ecefPosition, [Out] double[] geodeticLlh);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_geodetic_to_ecef", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_geodetic_to_ecef([In] double[] geodeticLlh, [Out] double[] ecefPosition);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_eci_to_geodetic", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_eci_to_geodetic([In] double[] eciPosition, double utcSeconds, [Out] double[] geodeticLlh);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_geodetic_to_eci", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_geodetic_to_eci([In] double[] geodeticLlh, double utcSeconds, [Out] double[] eciPosition);

        // RTN/ECI conversions
        [DllImport("j2_orbit_propagator", EntryPoint = "j2_rtn_to_eci_rotation", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_rtn_to_eci_rotation([In] double[] rEci, [In] double[] vEci, [Out] double[] RRowMajor9);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_eci_to_rtn_rotation", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_eci_to_rtn_rotation([In] double[] rEci, [In] double[] vEci, [Out] double[] RRowMajor9);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_eci_to_rtn_vector", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_eci_to_rtn_vector([In] double[] rEci, [In] double[] vEci, [In] double[] vecEci, [Out] double[] vecRtn);

        [DllImport("j2_orbit_propagator", EntryPoint = "j2_rtn_to_eci_vector", CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_rtn_to_eci_vector([In] double[] rEci, [In] double[] vEci, [In] double[] vecRtn, [Out] double[] vecEci);

        // === 星座传播器 C API 导入 ===

        [DllImport("j2_orbit_propagator", EntryPoint = "constellation_propagator_create", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr constellation_propagator_create(double epoch_time);

        [DllImport("j2_orbit_propagator", EntryPoint = "constellation_propagator_destroy", CallingConvention = CallingConvention.Cdecl)]
        public static extern void constellation_propagator_destroy(IntPtr handle);

        [DllImport("j2_orbit_propagator", EntryPoint = "constellation_propagator_add_satellites", CallingConvention = CallingConvention.Cdecl)]
        public static extern int constellation_propagator_add_satellites(IntPtr handle, [In] CCompactOrbitalElements[] satellites, nuint count);

        [DllImport("j2_orbit_propagator", EntryPoint = "constellation_propagator_add_satellite", CallingConvention = CallingConvention.Cdecl)]
        public static extern int constellation_propagator_add_satellite(IntPtr handle, ref CCompactOrbitalElements satellite);

        [DllImport("j2_orbit_propagator", EntryPoint = "constellation_propagator_get_satellite_count", CallingConvention = CallingConvention.Cdecl)]
        public static extern int constellation_propagator_get_satellite_count(IntPtr handle, out nuint count);

        [DllImport("j2_orbit_propagator", EntryPoint = "constellation_propagator_propagate", CallingConvention = CallingConvention.Cdecl)]
        public static extern int constellation_propagator_propagate(IntPtr handle, double target_time);

        [DllImport("j2_orbit_propagator", EntryPoint = "constellation_propagator_get_satellite_elements", CallingConvention = CallingConvention.Cdecl)]
        public static extern int constellation_propagator_get_satellite_elements(IntPtr handle, nuint satellite_id, out CCompactOrbitalElements elements);

        [DllImport("j2_orbit_propagator", EntryPoint = "constellation_propagator_get_satellite_state", CallingConvention = CallingConvention.Cdecl)]
        public static extern int constellation_propagator_get_satellite_state(IntPtr handle, nuint satellite_id, ref CStateVector state);

        [DllImport("j2_orbit_propagator", EntryPoint = "constellation_propagator_get_all_positions", CallingConvention = CallingConvention.Cdecl)]
        public static extern int constellation_propagator_get_all_positions(IntPtr handle, [Out] double[] positions, ref nuint count);

        [DllImport("j2_orbit_propagator", EntryPoint = "constellation_propagator_apply_impulse_to_constellation", CallingConvention = CallingConvention.Cdecl)]
        public static extern int constellation_propagator_apply_impulse_to_constellation(IntPtr handle, [In] double[] delta_vs, nuint count, double impulse_time);

        [DllImport("j2_orbit_propagator", EntryPoint = "constellation_propagator_apply_impulse_to_satellites", CallingConvention = CallingConvention.Cdecl)]
        public static extern int constellation_propagator_apply_impulse_to_satellites(IntPtr handle, [In] nuint[] satellite_ids, [In] double[] delta_vs, nuint count, double impulse_time);

        [DllImport("j2_orbit_propagator", EntryPoint = "constellation_propagator_set_step_size", CallingConvention = CallingConvention.Cdecl)]
        public static extern int constellation_propagator_set_step_size(IntPtr handle, double step_size);

        [DllImport("j2_orbit_propagator", EntryPoint = "constellation_propagator_set_compute_mode", CallingConvention = CallingConvention.Cdecl)]
        public static extern int constellation_propagator_set_compute_mode(IntPtr handle, ComputeMode mode);

        [DllImport("j2_orbit_propagator", EntryPoint = "constellation_propagator_set_adaptive_step_size", CallingConvention = CallingConvention.Cdecl)]
        public static extern int constellation_propagator_set_adaptive_step_size(IntPtr handle, int enable);

        [DllImport("j2_orbit_propagator", EntryPoint = "constellation_propagator_set_adaptive_parameters", CallingConvention = CallingConvention.Cdecl)]
        public static extern int constellation_propagator_set_adaptive_parameters(IntPtr handle, double tolerance, double min_step, double max_step);

        [DllImport("j2_orbit_propagator", EntryPoint = "constellation_propagator_is_cuda_available", CallingConvention = CallingConvention.Cdecl)]
        public static extern int constellation_propagator_is_cuda_available();
    }

    /// <summary>
    /// J2 摄动轨道传播器的 .NET 封装，提供传播、坐标转换、GMST 计算等便捷 API。
    /// </summary>
    public sealed class J2Orbit : IDisposable
    {
        private IntPtr _handle;

        /// <summary>
        /// 根据初始轨道根数创建一个传播器实例。
        /// </summary>
        /// <param name="initial">初始轨道根数。</param>
        /// <exception cref="NativeCallException">原生实例创建失败。</exception>
        public J2Orbit(COrbitalElements initial)
        {
            _handle = Native.j2_propagator_create(ref initial);
            if (_handle == IntPtr.Zero) throw new NativeCallException("Failed to create J2 propagator instance.");
        }

        /// <summary>
        /// 释放原生传播器句柄及其非托管资源。
        /// </summary>
        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                Native.j2_propagator_destroy(_handle);
                _handle = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }

        #pragma warning disable 1591
        /// <summary>
        /// 析构函数，确保在垃圾回收时释放非托管资源。
        /// </summary>
        ~J2Orbit() => Dispose();
        #pragma warning restore 1591

        /// <summary>
        /// 将轨道传播到目标时间（相对 J2000.0 的秒）。
        /// </summary>
        /// <param name="targetTime">目标时刻，单位秒（相对 J2000.0）。</param>
        /// <returns>目标时刻的轨道根数。</returns>
        /// <exception cref="PropagationException">传播失败。</exception>
        public COrbitalElements Propagate(double targetTime)
        {
            if (Native.j2_propagator_propagate(_handle, targetTime, out var result) != 0)
                throw new PropagationException("propagate failed");
            return result;
        }

        // 按UTC时刻传播（秒数基于J2000）
        /// <summary>
        /// 使用 UTC 时间传播至指定时刻。
        /// </summary>
        /// <param name="utc">UTC 时间。</param>
        /// <returns>目标 UTC 时刻的轨道根数。</returns>
        public COrbitalElements PropagateUtc(DateTime utc)
        {
            var t = TimeUtil.ToSecondsSinceJ2000(utc);
            return Propagate(t);
        }

        /// <summary>
        /// 由轨道根数计算位置-速度状态。
        /// </summary>
        /// <param name="elements">轨道根数。</param>
        /// <returns>ECI 位置-速度状态。</returns>
        /// <exception cref="ConversionException">转换失败。</exception>
        public CStateVector ElementsToState(COrbitalElements elements)
        {
            var state = CStateVector.Create();
            if (Native.j2_propagator_elements_to_state(_handle, ref elements, ref state) != 0)
                throw new ConversionException("elements_to_state failed");
            return state;
        }

        /// <summary>
        /// 由位置-速度状态反求轨道根数。
        /// </summary>
        /// <param name="state">ECI 位置-速度状态。</param>
        /// <param name="time">时刻（相对 J2000.0 秒）。</param>
        /// <returns>轨道根数。</returns>
        /// <exception cref="ConversionException">转换失败。</exception>
        public COrbitalElements StateToElements(CStateVector state, double time)
        {
            if (Native.j2_propagator_state_to_elements(_handle, ref state, time, out var elements) != 0)
                throw new ConversionException("state_to_elements failed");
            return elements;
        }

        /// <summary>
        /// 使用 UTC 时间由位置-速度状态反求轨道根数。
        /// </summary>
        /// <param name="state">ECI 位置-速度状态。</param>
        /// <param name="utc">UTC 时间。</param>
        /// <returns>轨道根数。</returns>
        public COrbitalElements StateToElementsUtc(CStateVector state, DateTime utc)
        {
            var t = TimeUtil.ToSecondsSinceJ2000(utc);
            return StateToElements(state, t);
        }

        /// <summary>
        /// 在给定时刻对航天器施加瞬时脉冲（ECI 坐标系），返回施加后轨道根数。
        /// </summary>
        /// <param name="elements">施加前轨道根数。</param>
        /// <param name="deltaV">脉冲 Δv（米/秒，长度为 3）。</param>
        /// <param name="t">时刻（相对 J2000.0 秒）。</param>
        /// <returns>施加脉冲后的轨道根数。</returns>
        /// <exception cref="ArgumentException">deltaV 长度不是 3。</exception>
        /// <exception cref="PropagationException">施加脉冲失败。</exception>
        public COrbitalElements ApplyImpulse(COrbitalElements elements, double[] deltaV, double t)
        {
            if (deltaV == null || deltaV.Length != 3) throw new ArgumentException("deltaV must be length 3");
            if (Native.j2_propagator_apply_impulse(_handle, ref elements, deltaV, t, out var result) != 0)
                throw new PropagationException("apply_impulse failed");
            return result;
        }

        /// <summary>
        /// 使用 UTC 时间施加瞬时脉冲（ECI 坐标系）。
        /// </summary>
        /// <param name="elements">施加前轨道根数。</param>
        /// <param name="deltaV">脉冲 Δv（米/秒，长度为 3）。</param>
        /// <param name="utc">UTC 时间。</param>
        /// <returns>施加脉冲后的轨道根数。</returns>
        public COrbitalElements ApplyImpulseUtc(COrbitalElements elements, double[] deltaV, DateTime utc)
        {
            var t = TimeUtil.ToSecondsSinceJ2000(utc);
            return ApplyImpulse(elements, deltaV, t);
        }

        /// <summary>
        /// 获取或设置传播积分步长（秒）。
        /// </summary>
        /// <exception cref="NativeCallException">获取或设置失败。</exception>
        public double StepSize
        {
            get { if (Native.j2_propagator_get_step_size(_handle, out var s) != 0) throw new NativeCallException("get_step_size failed"); return s; }
            set { if (Native.j2_propagator_set_step_size(_handle, value) != 0) throw new NativeCallException("set_step_size failed"); }
        }

        /// <summary>
        /// 启用或禁用自适应步长。
        /// </summary>
        /// <param name="enable">是否启用。</param>
        /// <exception cref="NativeCallException">设置失败。</exception>
        public void SetAdaptive(bool enable) { if (Native.j2_propagator_set_adaptive_step_size(_handle, enable ? 1 : 0) != 0) throw new NativeCallException("set_adaptive_step_size failed"); }

        /// <summary>
        /// 设置自适应步长参数。
        /// </summary>
        /// <param name="tol">误差容限。</param>
        /// <param name="minStep">最小步长（秒）。</param>
        /// <param name="maxStep">最大步长（秒）。</param>
        /// <exception cref="NativeCallException">设置失败。</exception>
        public void SetAdaptiveParameters(double tol, double minStep, double maxStep)
        { if (Native.j2_propagator_set_adaptive_parameters(_handle, tol, minStep, maxStep) != 0) throw new NativeCallException("set_adaptive_parameters failed"); }

        /// <summary>
        /// 将角度归一化到 (-π, π] 区间。
        /// </summary>
        /// <param name="angle">输入角度（弧度）。</param>
        /// <returns>归一化后的角度（弧度）。</returns>
        public static double NormalizeAngle(double angle) => Native.j2_normalize_angle(angle);

        // 便捷：ECI/ECEF与大地坐标封装
        /// <summary>
        /// 将 ECI 位置转换为 ECEF 位置。
        /// </summary>
        /// <param name="rEci">ECI 位置（米，长度为 3）。</param>
        /// <param name="utc">UTC 时间。</param>
        /// <returns>ECEF 位置（米，长度为 3）。</returns>
        /// <exception cref="ArgumentException">输入向量长度不是 3。</exception>
        /// <exception cref="ConversionException">转换失败。</exception>
        public static double[] EciToEcefPosition(double[] rEci, DateTime utc)
        {
            if (rEci == null || rEci.Length != 3) throw new ArgumentException("rEci must be length 3");
            var rEcef = new double[3];
            var t = TimeUtil.ToSecondsSinceJ2000(utc);
            if (Native.j2_eci_to_ecef_position(rEci, t, rEcef) != 0) throw new ConversionException("eci_to_ecef_position failed");
            return rEcef;
        }

        /// <summary>
        /// 将 ECEF 位置转换为 ECI 位置。
        /// </summary>
        /// <param name="rEcef">ECEF 位置（米，长度为 3）。</param>
        /// <param name="utc">UTC 时间。</param>
        /// <returns>ECI 位置（米，长度为 3）。</returns>
        /// <exception cref="ArgumentException">输入向量长度不是 3。</exception>
        /// <exception cref="ConversionException">转换失败。</exception>
        public static double[] EcefToEciPosition(double[] rEcef, DateTime utc)
        {
            if (rEcef == null || rEcef.Length != 3) throw new ArgumentException("rEcef must be length 3");
            var rEci = new double[3];
            var t = TimeUtil.ToSecondsSinceJ2000(utc);
            if (Native.j2_ecef_to_eci_position(rEcef, t, rEci) != 0) throw new ConversionException("ecef_to_eci_position failed");
            return rEci;
        }

        /// <summary>
        /// 将 ECI 速度转换为 ECEF 速度。
        /// </summary>
        /// <param name="rEci">ECI 位置（米，长度为 3）。</param>
        /// <param name="vEci">ECI 速度（米/秒，长度为 3）。</param>
        /// <param name="utc">UTC 时间。</param>
        /// <returns>ECEF 速度（米/秒，长度为 3）。</returns>
        /// <exception cref="ArgumentException">输入向量长度不是 3。</exception>
        /// <exception cref="ConversionException">转换失败。</exception>
        public static double[] EciToEcefVelocity(double[] rEci, double[] vEci, DateTime utc)
        {
            if (rEci == null || rEci.Length != 3) throw new ArgumentException("rEci must be length 3");
            if (vEci == null || vEci.Length != 3) throw new ArgumentException("vEci must be length 3");
            var vEcef = new double[3];
            var t = TimeUtil.ToSecondsSinceJ2000(utc);
            if (Native.j2_eci_to_ecef_velocity(rEci, vEci, t, vEcef) != 0) throw new ConversionException("eci_to_ecef_velocity failed");
            return vEcef;
        }

        /// <summary>
        /// 将 ECEF 速度转换为 ECI 速度。
        /// </summary>
        /// <param name="rEcef">ECEF 位置（米，长度为 3）。</param>
        /// <param name="vEcef">ECEF 速度（米/秒，长度为 3）。</param>
        /// <param name="utc">UTC 时间。</param>
        /// <returns>ECI 速度（米/秒，长度为 3）。</returns>
        /// <exception cref="ArgumentException">输入向量长度不是 3。</exception>
        /// <exception cref="ConversionException">转换失败。</exception>
        public static double[] EcefToEciVelocity(double[] rEcef, double[] vEcef, DateTime utc)
        {
            if (rEcef == null || rEcef.Length != 3) throw new ArgumentException("rEcef must be length 3");
            if (vEcef == null || vEcef.Length != 3) throw new ArgumentException("vEcef must be length 3");
            var vEci = new double[3];
            var t = TimeUtil.ToSecondsSinceJ2000(utc);
            if (Native.j2_ecef_to_eci_velocity(rEcef, vEcef, t, vEci) != 0) throw new ConversionException("ecef_to_eci_velocity failed");
            return vEci;
        }

        // 便捷：ECI -> Geodetic
        /// <summary>
        /// 将 ECI 位置转换为大地坐标（经纬高）。
        /// </summary>
        /// <param name="rEci">ECI 位置（米，长度为 3）。</param>
        /// <param name="utc">UTC 时间。</param>
        /// <returns>大地坐标（弧度经纬，高程米）。</returns>
        public static GeodeticCoord EciToGeodetic(double[] rEci, DateTime utc)
        {
            if (rEci == null || rEci.Length != 3) throw new ArgumentException("rEci must be length 3");
            var llh = new double[3];
            var t = TimeUtil.ToSecondsSinceJ2000(utc);
            if (Native.j2_eci_to_geodetic(rEci, t, llh) != 0) throw new ConversionException("eci_to_geodetic failed");
            return new GeodeticCoord { LatRad = llh[0], LonRad = llh[1], AltMeters = llh[2] };
        }

        // 便捷：Geodetic -> ECI
        /// <summary>
        /// 将大地坐标（经纬高）转换为 ECI 位置。
        /// </summary>
        /// <param name="geo">大地坐标（弧度经纬，高程米）。</param>
        /// <param name="utc">UTC 时间。</param>
        /// <returns>ECI 位置（米，长度为 3）。</returns>
        public static double[] GeodeticToEci(GeodeticCoord geo, DateTime utc)
        {
            var llh = new[] { geo.LatRad, geo.LonRad, geo.AltMeters };
            var rEci = new double[3];
            var t = TimeUtil.ToSecondsSinceJ2000(utc);
            if (Native.j2_geodetic_to_eci(llh, t, rEci) != 0) throw new ConversionException("geodetic_to_eci failed");
            return rEci;
        }

        /// <summary>
        /// 计算指定 UTC 时间的格林尼治平恒星时（GMST）。
        /// </summary>
        /// <param name="utc">UTC 时间。</param>
        /// <param name="gmst">返回的 GMST（弧度）。</param>
        /// <returns>0 表示成功，非 0 表示失败。</returns>
        public static int ComputeGmstUtc(DateTime utc, out double gmst)
        {
            var t = TimeUtil.ToSecondsSinceJ2000(utc);
            return Native.j2_compute_gmst(t, out gmst);
        }
    }

    /// <summary>
    /// 星座传播器的 .NET 封装，支持批量卫星传播、SIMD/CUDA 加速等功能。
    /// </summary>
    public sealed class ConstellationPropagator : IDisposable
    {
        private IntPtr _handle;

        /// <summary>
        /// 根据历元时间创建星座传播器实例。
        /// </summary>
        /// <param name="epochTime">星座统一历元时间（相对 J2000.0 秒）。</param>
        /// <exception cref="NativeCallException">原生实例创建失败。</exception>
        public ConstellationPropagator(double epochTime)
        {
            _handle = Native.constellation_propagator_create(epochTime);
            if (_handle == IntPtr.Zero) throw new NativeCallException("Failed to create constellation propagator instance.");
        }

        /// <summary>
        /// 使用 UTC 时间创建星座传播器实例。
        /// </summary>
        /// <param name="epochUtc">星座统一历元 UTC 时间。</param>
        /// <exception cref="NativeCallException">原生实例创建失败。</exception>
        public ConstellationPropagator(DateTime epochUtc) : this(TimeUtil.ToSecondsSinceJ2000(epochUtc))
        {
        }

        /// <summary>
        /// 释放原生传播器句柄及其非托管资源。
        /// </summary>
        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                Native.constellation_propagator_destroy(_handle);
                _handle = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }

        #pragma warning disable 1591
        /// <summary>
        /// 析构函数，确保在垃圾回收时释放非托管资源。
        /// </summary>
        ~ConstellationPropagator() => Dispose();
        #pragma warning restore 1591

        /// <summary>
        /// 批量添加卫星到星座。
        /// </summary>
        /// <param name="satellites">卫星轨道要素数组。</param>
        /// <exception cref="ArgumentException">satellites 为空或长度为 0。</exception>
        /// <exception cref="NativeCallException">添加失败。</exception>
        public void AddSatellites(CCompactOrbitalElements[] satellites)
        {
            if (satellites == null || satellites.Length == 0) throw new ArgumentException("satellites cannot be null or empty");
            if (Native.constellation_propagator_add_satellites(_handle, satellites, (nuint)satellites.Length) != 0)
                throw new NativeCallException("Failed to add satellites");
        }

        /// <summary>
        /// 添加单个卫星到星座。
        /// </summary>
        /// <param name="satellite">卫星轨道要素。</param>
        /// <exception cref="NativeCallException">添加失败。</exception>
        public void AddSatellite(CCompactOrbitalElements satellite)
        {
            if (Native.constellation_propagator_add_satellite(_handle, ref satellite) != 0)
                throw new NativeCallException("Failed to add satellite");
        }

        /// <summary>
        /// 获取星座中的卫星数量。
        /// </summary>
        /// <returns>卫星数量。</returns>
        /// <exception cref="NativeCallException">获取失败。</exception>
        public int GetSatelliteCount()
        {
            if (Native.constellation_propagator_get_satellite_count(_handle, out var count) != 0)
                throw new NativeCallException("Failed to get satellite count");
            return (int)count;
        }

        /// <summary>
        /// 将整个星座传播到指定时间。
        /// </summary>
        /// <param name="targetTime">目标时间（相对 J2000.0 秒）。</param>
        /// <exception cref="PropagationException">传播失败。</exception>
        public void Propagate(double targetTime)
        {
            if (Native.constellation_propagator_propagate(_handle, targetTime) != 0)
                throw new PropagationException("Failed to propagate constellation");
        }

        /// <summary>
        /// 使用 UTC 时间将整个星座传播到指定时间。
        /// </summary>
        /// <param name="targetUtc">目标 UTC 时间。</param>
        /// <exception cref="PropagationException">传播失败。</exception>
        public void PropagateUtc(DateTime targetUtc)
        {
            var t = TimeUtil.ToSecondsSinceJ2000(targetUtc);
            Propagate(t);
        }

        /// <summary>
        /// 获取指定卫星的当前轨道要素。
        /// </summary>
        /// <param name="satelliteId">卫星索引（从 0 开始）。</param>
        /// <returns>轨道要素。</returns>
        /// <exception cref="ArgumentOutOfRangeException">卫星索引超出范围。</exception>
        /// <exception cref="NativeCallException">获取失败。</exception>
        public CCompactOrbitalElements GetSatelliteElements(int satelliteId)
        {
            if (satelliteId < 0) throw new ArgumentOutOfRangeException(nameof(satelliteId), "satelliteId cannot be negative");
            if (Native.constellation_propagator_get_satellite_elements(_handle, (nuint)satelliteId, out var elements) != 0)
                throw new NativeCallException("Failed to get satellite elements");
            return elements;
        }

        /// <summary>
        /// 获取指定卫星的当前状态向量。
        /// </summary>
        /// <param name="satelliteId">卫星索引（从 0 开始）。</param>
        /// <returns>状态向量。</returns>
        /// <exception cref="ArgumentOutOfRangeException">卫星索引超出范围。</exception>
        /// <exception cref="NativeCallException">获取失败。</exception>
        public CStateVector GetSatelliteState(int satelliteId)
        {
            if (satelliteId < 0) throw new ArgumentOutOfRangeException(nameof(satelliteId), "satelliteId cannot be negative");
            var state = CStateVector.Create();
            if (Native.constellation_propagator_get_satellite_state(_handle, (nuint)satelliteId, ref state) != 0)
                throw new NativeCallException("Failed to get satellite state");
            return state;
        }

        /// <summary>
        /// 获取所有卫星的位置。
        /// </summary>
        /// <returns>位置数组，每个卫星 3 个坐标 [x1,y1,z1,x2,y2,z2,...]。</returns>
        /// <exception cref="NativeCallException">获取失败。</exception>
        public double[] GetAllPositions()
        {
            var count = (nuint)GetSatelliteCount();
            var positions = new double[(int)count * 3];
            if (Native.constellation_propagator_get_all_positions(_handle, positions, ref count) != 0)
                throw new NativeCallException("Failed to get all positions");
            return positions;
        }

        /// <summary>
        /// 对整个星座施加脉冲。
        /// </summary>
        /// <param name="deltaVs">速度增量数组 [dvx1,dvy1,dvz1,dvx2,dvy2,dvz2,...]，需要 3*卫星数量 个元素。</param>
        /// <param name="impulseTime">脉冲施加时间（相对 J2000.0 秒）。</param>
        /// <exception cref="ArgumentException">deltaVs 长度不正确。</exception>
        /// <exception cref="PropagationException">施加脉冲失败。</exception>
        public void ApplyImpulseToConstellation(double[] deltaVs, double impulseTime)
        {
            var satCount = GetSatelliteCount();
            if (deltaVs == null || deltaVs.Length != satCount * 3)
                throw new ArgumentException($"deltaVs must have length {satCount * 3} (3 * satellite count)");
            if (Native.constellation_propagator_apply_impulse_to_constellation(_handle, deltaVs, (nuint)satCount, impulseTime) != 0)
                throw new PropagationException("Failed to apply impulse to constellation");
        }

        /// <summary>
        /// 使用 UTC 时间对整个星座施加脉冲。
        /// </summary>
        /// <param name="deltaVs">速度增量数组 [dvx1,dvy1,dvz1,dvx2,dvy2,dvz2,...]。</param>
        /// <param name="impulseUtc">脉冲施加 UTC 时间。</param>
        /// <exception cref="ArgumentException">deltaVs 长度不正确。</exception>
        /// <exception cref="PropagationException">施加脉冲失败。</exception>
        public void ApplyImpulseToConstellationUtc(double[] deltaVs, DateTime impulseUtc)
        {
            var t = TimeUtil.ToSecondsSinceJ2000(impulseUtc);
            ApplyImpulseToConstellation(deltaVs, t);
        }

        /// <summary>
        /// 对指定卫星子集施加脉冲。
        /// </summary>
        /// <param name="satelliteIds">卫星索引数组。</param>
        /// <param name="deltaVs">速度增量数组 [dvx1,dvy1,dvz1,dvx2,dvy2,dvz2,...]，需要 3*satelliteIds.Length 个元素。</param>
        /// <param name="impulseTime">脉冲施加时间（相对 J2000.0 秒）。</param>
        /// <exception cref="ArgumentException">参数为空或长度不匹配。</exception>
        /// <exception cref="PropagationException">施加脉冲失败。</exception>
        public void ApplyImpulseToSatellites(int[] satelliteIds, double[] deltaVs, double impulseTime)
        {
            if (satelliteIds == null || satelliteIds.Length == 0) throw new ArgumentException("satelliteIds cannot be null or empty");
            if (deltaVs == null || deltaVs.Length != satelliteIds.Length * 3)
                throw new ArgumentException($"deltaVs must have length {satelliteIds.Length * 3} (3 * satelliteIds count)");

            var satelliteIdsPtrs = Array.ConvertAll(satelliteIds, id => (nuint)id);
            if (Native.constellation_propagator_apply_impulse_to_satellites(_handle, satelliteIdsPtrs, deltaVs, (nuint)satelliteIds.Length, impulseTime) != 0)
                throw new PropagationException("Failed to apply impulse to satellites");
        }

        /// <summary>
        /// 使用 UTC 时间对指定卫星子集施加脉冲。
        /// </summary>
        /// <param name="satelliteIds">卫星索引数组。</param>
        /// <param name="deltaVs">速度增量数组 [dvx1,dvy1,dvz1,dvx2,dvy2,dvz2,...]。</param>
        /// <param name="impulseUtc">脉冲施加 UTC 时间。</param>
        /// <exception cref="ArgumentException">参数为空或长度不匹配。</exception>
        /// <exception cref="PropagationException">施加脉冲失败。</exception>
        public void ApplyImpulseToSatellitesUtc(int[] satelliteIds, double[] deltaVs, DateTime impulseUtc)
        {
            var t = TimeUtil.ToSecondsSinceJ2000(impulseUtc);
            ApplyImpulseToSatellites(satelliteIds, deltaVs, t);
        }

        /// <summary>
        /// 设置积分步长。
        /// </summary>
        /// <param name="stepSize">步长（秒）。</param>
        /// <exception cref="NativeCallException">设置失败。</exception>
        public void SetStepSize(double stepSize)
        {
            if (Native.constellation_propagator_set_step_size(_handle, stepSize) != 0)
                throw new NativeCallException("Failed to set step size");
        }

        /// <summary>
        /// 设置计算模式。
        /// </summary>
        /// <param name="mode">计算模式。</param>
        /// <exception cref="NativeCallException">设置失败。</exception>
        public void SetComputeMode(ComputeMode mode)
        {
            if (Native.constellation_propagator_set_compute_mode(_handle, mode) != 0)
                throw new NativeCallException("Failed to set compute mode");
        }

        /// <summary>
        /// 启用或禁用自适应步长。
        /// </summary>
        /// <param name="enable">是否启用。</param>
        /// <exception cref="NativeCallException">设置失败。</exception>
        public void SetAdaptive(bool enable)
        {
            if (Native.constellation_propagator_set_adaptive_step_size(_handle, enable ? 1 : 0) != 0)
                throw new NativeCallException("Failed to set adaptive step size");
        }

        /// <summary>
        /// 设置自适应步长参数。
        /// </summary>
        /// <param name="tolerance">误差容忍度。</param>
        /// <param name="minStep">最小步长（秒）。</param>
        /// <param name="maxStep">最大步长（秒）。</param>
        /// <exception cref="NativeCallException">设置失败。</exception>
        public void SetAdaptiveParameters(double tolerance, double minStep, double maxStep)
        {
            if (Native.constellation_propagator_set_adaptive_parameters(_handle, tolerance, minStep, maxStep) != 0)
                throw new NativeCallException("Failed to set adaptive parameters");
        }

        /// <summary>
        /// 检查 CUDA 是否可用。
        /// </summary>
        /// <returns>CUDA 是否可用。</returns>
        public static bool IsCudaAvailable()
        {
            return Native.constellation_propagator_is_cuda_available() != 0;
        }
    }
}