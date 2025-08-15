using System;
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

    internal static class Native
    {
        // 使用跨平台库基名，运行时自动映射到 .dll/.so/.dylib
        public const string DllName = "j2_orbit_propagator";
    }

    public sealed class J2Orbit : IDisposable
    {
        private IntPtr _handle;

        [DllImport(Native.DllName, EntryPoint = "j2_propagator_create", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr j2_propagator_create(ref COrbitalElements elements);

        [DllImport(Native.DllName, EntryPoint = "j2_propagator_destroy", CallingConvention = CallingConvention.Cdecl)]
        private static extern void j2_propagator_destroy(IntPtr handle);

        [DllImport(Native.DllName, EntryPoint = "j2_propagator_elements_to_state", CallingConvention = CallingConvention.Cdecl)]
        private static extern int j2_propagator_elements_to_state(IntPtr handle, ref COrbitalElements elements, ref CStateVector state);

        [DllImport(Native.DllName, EntryPoint = "j2_propagator_state_to_elements", CallingConvention = CallingConvention.Cdecl)]
        private static extern int j2_propagator_state_to_elements(IntPtr handle, ref CStateVector state, double time, out COrbitalElements elements);

        public J2Orbit(COrbitalElements elems)
        {
            _handle = j2_propagator_create(ref elems);
            if (_handle == IntPtr.Zero) throw new PropagationException("Failed to create J2Orbit propagator");
        }

        public CStateVector ElementsToState(COrbitalElements elems)
        {
            var state = CStateVector.Create();
            if (j2_propagator_elements_to_state(_handle, ref elems, ref state) != 0)
                throw new ConversionException("elements_to_state failed");
            return state;
        }

        public COrbitalElements StateToElements(CStateVector state, double t)
        {
            if (j2_propagator_state_to_elements(_handle, ref state, t, out var elements) != 0)
                throw new ConversionException("state_to_elements failed");
            return elements;
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                j2_propagator_destroy(_handle);
                _handle = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }

        ~J2Orbit() { Dispose(); }
    }

    public sealed class ConstellationPropagator : IDisposable
    {
        private IntPtr _handle;

        // P/Invoke methods
        [DllImport(Native.DllName, EntryPoint = "constellation_propagator_create", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr constellation_propagator_create(double epoch_time);

        [DllImport(Native.DllName, EntryPoint = "constellation_propagator_destroy", CallingConvention = CallingConvention.Cdecl)]
        private static extern void constellation_propagator_destroy(IntPtr handle);

        [DllImport(Native.DllName, EntryPoint = "constellation_propagator_add_satellite", CallingConvention = CallingConvention.Cdecl)]
        private static extern int constellation_propagator_add_satellite(IntPtr handle, ref CCompactOrbitalElements elems);

        [DllImport(Native.DllName, EntryPoint = "constellation_propagator_add_satellites", CallingConvention = CallingConvention.Cdecl)]
        private static extern int constellation_propagator_add_satellites(IntPtr handle, [In] CCompactOrbitalElements[] elems, nuint count);

        [DllImport(Native.DllName, EntryPoint = "constellation_propagator_is_cuda_available", CallingConvention = CallingConvention.Cdecl)]
        private static extern int constellation_propagator_is_cuda_available();

        [DllImport(Native.DllName, EntryPoint = "constellation_propagator_propagate", CallingConvention = CallingConvention.Cdecl)]
        private static extern int constellation_propagator_propagate(IntPtr handle, double target_time);

        [DllImport(Native.DllName, EntryPoint = "constellation_propagator_get_satellite_count", CallingConvention = CallingConvention.Cdecl)]
        private static extern int constellation_propagator_get_satellite_count(IntPtr handle, out nuint count);

        [DllImport(Native.DllName, EntryPoint = "constellation_propagator_get_satellite_elements", CallingConvention = CallingConvention.Cdecl)]
        private static extern int constellation_propagator_get_satellite_elements(IntPtr handle, nuint satellite_id, out CCompactOrbitalElements elements);

        [DllImport(Native.DllName, EntryPoint = "constellation_propagator_get_satellite_state", CallingConvention = CallingConvention.Cdecl)]
        private static extern int constellation_propagator_get_satellite_state(IntPtr handle, nuint satellite_id, ref CStateVector state);

        [DllImport(Native.DllName, EntryPoint = "constellation_propagator_get_all_positions", CallingConvention = CallingConvention.Cdecl)]
        private static extern int constellation_propagator_get_all_positions(IntPtr handle, [Out] double[] positions, ref nuint count);

        [DllImport(Native.DllName, EntryPoint = "constellation_propagator_apply_impulse_to_constellation", CallingConvention = CallingConvention.Cdecl)]
        private static extern int constellation_propagator_apply_impulse_to_constellation(IntPtr handle, [In] double[] deltaVs, nuint count, double impulse_time);

        // Additional P/Invoke methods for configuration
        [DllImport(Native.DllName, EntryPoint = "constellation_propagator_set_step_size", CallingConvention = CallingConvention.Cdecl)]
        private static extern int constellation_propagator_set_step_size(IntPtr handle, double step_size);

        [DllImport(Native.DllName, EntryPoint = "constellation_propagator_set_compute_mode", CallingConvention = CallingConvention.Cdecl)]
        private static extern int constellation_propagator_set_compute_mode(IntPtr handle, ComputeMode mode);

        [DllImport(Native.DllName, EntryPoint = "constellation_propagator_set_adaptive_step_size", CallingConvention = CallingConvention.Cdecl)]
        private static extern int constellation_propagator_set_adaptive_step_size(IntPtr handle, int enable);

        [DllImport(Native.DllName, EntryPoint = "constellation_propagator_set_adaptive_parameters", CallingConvention = CallingConvention.Cdecl)]
        private static extern int constellation_propagator_set_adaptive_parameters(IntPtr handle, double tolerance, double min_step, double max_step);

        // Public API
        public static bool IsCudaAvailable() => constellation_propagator_is_cuda_available() != 0;

        public ConstellationPropagator(double epochTimeSeconds)
        {
            _handle = constellation_propagator_create(epochTimeSeconds);
            if (_handle == IntPtr.Zero) throw new PropagationException("Failed to create ConstellationPropagator");
            
            // 如果 CUDA 可用，自动优先使用 CUDA 计算模式
            if (IsCudaAvailable())
            {
                SetComputeMode(ComputeMode.GpuCuda);
            }
        }

        public void AddSatellite(CCompactOrbitalElements elems)
        {
            if (constellation_propagator_add_satellite(_handle, ref elems) != 0)
                throw new PropagationException("Failed to add satellite");
        }

        public void AddSatellites(CCompactOrbitalElements[] elems)
        {
            if (constellation_propagator_add_satellites(_handle, elems, (nuint)elems.Length) != 0)
                throw new PropagationException("Failed to add satellites");
        }

        public void Propagate(double targetTimeSeconds)
        {
            if (constellation_propagator_propagate(_handle, targetTimeSeconds) != 0)
                throw new PropagationException("Failed to propagate constellation");
        }

        public int GetSatelliteCount()
        {
            if (constellation_propagator_get_satellite_count(_handle, out var count) != 0)
                throw new PropagationException("Failed to get satellite count");
            return (int)count;
        }

        public CCompactOrbitalElements GetSatelliteElements(int index)
        {
            if (constellation_propagator_get_satellite_elements(_handle, (nuint)index, out var elements) != 0)
                throw new PropagationException("Failed to get satellite elements");
            return elements;
        }

        public CStateVector GetSatelliteState(int index)
        {
            var state = CStateVector.Create();
            if (constellation_propagator_get_satellite_state(_handle, (nuint)index, ref state) != 0)
                throw new PropagationException("Failed to get satellite state");
            return state;
        }

        public double[] GetAllPositions()
        {
            var count = (nuint)GetSatelliteCount();
            var positions = new double[count * 3];
            var actualCount = count;
            if (constellation_propagator_get_all_positions(_handle, positions, ref actualCount) != 0)
                throw new PropagationException("Failed to get all positions");
            
            // Resize array if needed
            if (actualCount != count)
            {
                var result = new double[actualCount * 3];
                Array.Copy(positions, result, (int)(actualCount * 3));
                return result;
            }
            return positions;
        }

        public void ApplyImpulseToConstellation(double[] deltaVs, double targetTimeSeconds)
        {
            if (deltaVs == null) throw new ArgumentNullException(nameof(deltaVs));
            int satCount = GetSatelliteCount();
            if (deltaVs.Length != satCount * 3)
                throw new ArgumentException($"deltaVs length must be 3 * satelliteCount (expected {satCount * 3}, got {deltaVs.Length})", nameof(deltaVs));
            if (constellation_propagator_apply_impulse_to_constellation(_handle, deltaVs, (nuint)satCount, targetTimeSeconds) != 0)
                throw new PropagationException("Failed to apply impulse to constellation");
        }

        // Configuration methods
        public void SetStepSize(double stepSize)
        {
            if (constellation_propagator_set_step_size(_handle, stepSize) != 0)
                throw new PropagationException("Failed to set step size");
        }

        public void SetComputeMode(ComputeMode mode)
        {
            if (constellation_propagator_set_compute_mode(_handle, mode) != 0)
                throw new PropagationException("Failed to set compute mode");
        }

        public void SetAdaptive(bool enable)
        {
            if (constellation_propagator_set_adaptive_step_size(_handle, enable ? 1 : 0) != 0)
                throw new PropagationException("Failed to set adaptive step size");
        }

        public void SetAdaptiveParameters(double tolerance, double minStep, double maxStep)
        {
            if (constellation_propagator_set_adaptive_parameters(_handle, tolerance, minStep, maxStep) != 0)
                throw new PropagationException("Failed to set adaptive parameters");
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                constellation_propagator_destroy(_handle);
                _handle = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }

        ~ConstellationPropagator() { Dispose(); }
    }
}