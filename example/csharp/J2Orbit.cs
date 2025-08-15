using System;
using System.Runtime.InteropServices;

namespace J2.Propagator
{
    // 具体异常类型映射
    public class NativeCallException : Exception { public NativeCallException(string msg) : base(msg) {} }
    public sealed class PropagationException : NativeCallException { public PropagationException(string msg) : base(msg) {} }
    public sealed class ConversionException : NativeCallException { public ConversionException(string msg) : base(msg) {} }

    // 与C结构体内存布局保持一致
    [StructLayout(LayoutKind.Sequential)]
    public struct COrbitalElements
    {
        public double a;
        public double e;
        public double i;
        public double O;
        public double w;
        public double M;
        public double t;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CStateVector
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public double[] r;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public double[] v;

        public static CStateVector Create()
        {
            return new CStateVector { r = new double[3], v = new double[3] };
        }
    }

    // 地理经纬高（弧度 + 米），提供度制便捷属性
    public struct GeodeticCoord
    {
        public double LatRad;   // 纬度 (rad)
        public double LonRad;   // 经度 (rad)
        public double AltMeters; // 高程 (m)

        public double LatDeg { get => LatRad * 180.0 / Math.PI; set => LatRad = value * Math.PI / 180.0; }
        public double LonDeg { get => LonRad * 180.0 / Math.PI; set => LonRad = value * Math.PI / 180.0; }

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
    public static class TimeUtil
    {
        public static readonly DateTime J2000EpochUtc = new DateTime(2000, 1, 1, 12, 0, 0, DateTimeKind.Utc);
        public static double ToSecondsSinceJ2000(DateTime utc)
        {
            if (utc.Kind != DateTimeKind.Utc) utc = utc.ToUniversalTime();
            return (utc - J2000EpochUtc).TotalSeconds;
        }
        public static DateTime FromSecondsSinceJ2000(double seconds)
            => J2000EpochUtc.AddSeconds(seconds);
    }

    // 地理坐标与ECEF互转
    public static class GeoConversion
    {
        // ECEF -> 大地坐标（Bowring公式）
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

    internal static class Native
    {
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
    }

    public sealed class J2Orbit : IDisposable
    {
        private IntPtr _handle;

        public J2Orbit(COrbitalElements initial)
        {
            _handle = Native.j2_propagator_create(ref initial);
            if (_handle == IntPtr.Zero) throw new NativeCallException("Failed to create J2 propagator instance.");
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                Native.j2_propagator_destroy(_handle);
                _handle = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }

        ~J2Orbit() => Dispose();

        public COrbitalElements Propagate(double targetTime)
        {
            if (Native.j2_propagator_propagate(_handle, targetTime, out var result) != 0)
                throw new PropagationException("propagate failed");
            return result;
        }

        // 按UTC时刻传播（秒数基于J2000）
        public COrbitalElements PropagateUtc(DateTime utc)
        {
            var t = TimeUtil.ToSecondsSinceJ2000(utc);
            return Propagate(t);
        }

        public CStateVector ElementsToState(COrbitalElements elements)
        {
            var state = CStateVector.Create();
            if (Native.j2_propagator_elements_to_state(_handle, ref elements, ref state) != 0)
                throw new ConversionException("elements_to_state failed");
            return state;
        }

        public COrbitalElements StateToElements(CStateVector state, double time)
        {
            if (Native.j2_propagator_state_to_elements(_handle, ref state, time, out var elements) != 0)
                throw new ConversionException("state_to_elements failed");
            return elements;
        }

        public COrbitalElements StateToElementsUtc(CStateVector state, DateTime utc)
        {
            var t = TimeUtil.ToSecondsSinceJ2000(utc);
            return StateToElements(state, t);
        }

        public COrbitalElements ApplyImpulse(COrbitalElements elements, double[] deltaV, double t)
        {
            if (deltaV == null || deltaV.Length != 3) throw new ArgumentException("deltaV must be length 3");
            if (Native.j2_propagator_apply_impulse(_handle, ref elements, deltaV, t, out var result) != 0)
                throw new PropagationException("apply_impulse failed");
            return result;
        }

        public COrbitalElements ApplyImpulseUtc(COrbitalElements elements, double[] deltaV, DateTime utc)
        {
            var t = TimeUtil.ToSecondsSinceJ2000(utc);
            return ApplyImpulse(elements, deltaV, t);
        }

        public double StepSize
        {
            get { if (Native.j2_propagator_get_step_size(_handle, out var s) != 0) throw new NativeCallException("get_step_size failed"); return s; }
            set { if (Native.j2_propagator_set_step_size(_handle, value) != 0) throw new NativeCallException("set_step_size failed"); }
        }

        public void SetAdaptive(bool enable) { if (Native.j2_propagator_set_adaptive_step_size(_handle, enable ? 1 : 0) != 0) throw new NativeCallException("set_adaptive_step_size failed"); }

        public void SetAdaptiveParameters(double tol, double minStep, double maxStep)
        { if (Native.j2_propagator_set_adaptive_parameters(_handle, tol, minStep, maxStep) != 0) throw new NativeCallException("set_adaptive_parameters failed"); }

        public static double NormalizeAngle(double angle) => Native.j2_normalize_angle(angle);

        // 便捷：ECI/ECEF与大地坐标封装
        public static double[] EciToEcefPosition(double[] rEci, DateTime utc)
        {
            if (rEci == null || rEci.Length != 3) throw new ArgumentException("rEci must be length 3");
            var rEcef = new double[3];
            var t = TimeUtil.ToSecondsSinceJ2000(utc);
            if (Native.j2_eci_to_ecef_position(rEci, t, rEcef) != 0) throw new ConversionException("eci_to_ecef_position failed");
            return rEcef;
        }

        public static double[] EcefToEciPosition(double[] rEcef, DateTime utc)
        {
            if (rEcef == null || rEcef.Length != 3) throw new ArgumentException("rEcef must be length 3");
            var rEci = new double[3];
            var t = TimeUtil.ToSecondsSinceJ2000(utc);
            if (Native.j2_ecef_to_eci_position(rEcef, t, rEci) != 0) throw new ConversionException("ecef_to_eci_position failed");
            return rEci;
        }

        public static double[] EciToEcefVelocity(double[] rEci, double[] vEci, DateTime utc)
        {
            if (rEci == null || rEci.Length != 3) throw new ArgumentException("rEci must be length 3");
            if (vEci == null || vEci.Length != 3) throw new ArgumentException("vEci must be length 3");
            var vEcef = new double[3];
            var t = TimeUtil.ToSecondsSinceJ2000(utc);
            if (Native.j2_eci_to_ecef_velocity(rEci, vEci, t, vEcef) != 0) throw new ConversionException("eci_to_ecef_velocity failed");
            return vEcef;
        }

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
        public static GeodeticCoord EciToGeodetic(double[] rEci, DateTime utc)
        {
            var rEcef = EciToEcefPosition(rEci, utc);
            return GeoConversion.EcefToGeodetic(rEcef[0], rEcef[1], rEcef[2]);
        }

        // 便捷：Geodetic -> ECI
        public static double[] GeodeticToEci(GeodeticCoord geo, DateTime utc)
        {
            var (x, y, z) = GeoConversion.GeodeticToEcef(geo);
            var rEcef = new[] { x, y, z };
            return EcefToEciPosition(rEcef, utc);
        }

        public static int ComputeGmstUtc(DateTime utc, out double gmst)
        {
            var t = TimeUtil.ToSecondsSinceJ2000(utc);
            return Native.j2_compute_gmst(t, out gmst);
        }
    }
}