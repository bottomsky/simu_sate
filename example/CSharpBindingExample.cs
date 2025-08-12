using System;
using System.Runtime.InteropServices;

namespace J2OrbitPropagatorBinding
{
    /// <summary>
    /// C格式的轨道要素结构体
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct COrbitalElements
    {
        public double a;   // 半长轴 (m)
        public double e;   // 偏心率
        public double i;   // 倾角 (rad)
        public double O;   // 升交点赤经 (rad)
        public double w;   // 近地点幅角 (rad)
        public double M;   // 平近点角 (rad)
        public double t;   // 历元时间 (s)

        public override string ToString()
        {
            return $"COrbitalElements(a={a:F3}, e={e:F6}, i={i:F6}, O={O:F6}, w={w:F6}, M={M:F6}, t={t:F3})";
        }
    }

    /// <summary>
    /// C格式的状态向量结构体
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CStateVector
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public double[] r;  // 位置矢量 (m)
        
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public double[] v;  // 速度矢量 (m/s)

        public CStateVector(double[] position, double[] velocity)
        {
            r = new double[3];
            v = new double[3];
            Array.Copy(position, r, 3);
            Array.Copy(velocity, v, 3);
        }

        public override string ToString()
        {
            return $"CStateVector(r=[{r[0]:F3}, {r[1]:F3}, {r[2]:F3}], v=[{v[0]:F3}, {v[1]:F3}, {v[2]:F3}])";
        }
    }

    /// <summary>
    /// J2轨道传播器的P/Invoke声明
    /// </summary>
    public static class J2OrbitPropagatorNative
    {
        private const string DllName = "j2_orbit_propagator";

        // === 创建和销毁函数 ===
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr j2_propagator_create(ref COrbitalElements initial_elements);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void j2_propagator_destroy(IntPtr handle);

        // === 轨道传播函数 ===
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_propagator_propagate(IntPtr handle, double target_time, out COrbitalElements result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_propagator_elements_to_state(IntPtr handle, ref COrbitalElements elements, out CStateVector state);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_propagator_state_to_elements(IntPtr handle, ref CStateVector state, double time, out COrbitalElements elements);

        // === 参数设置函数 ===
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_propagator_set_step_size(IntPtr handle, double step_size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_propagator_get_step_size(IntPtr handle, out double step_size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_propagator_set_adaptive_step_size(IntPtr handle, int enable);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_propagator_set_adaptive_parameters(IntPtr handle, double tolerance, double min_step, double max_step);

        // === 坐标转换函数 ===
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_eci_to_ecef_position([In] double[] eci_position, double utc_seconds, [Out] double[] ecef_position);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_ecef_to_eci_position([In] double[] ecef_position, double utc_seconds, [Out] double[] eci_position);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_eci_to_ecef_velocity([In] double[] eci_position, [In] double[] eci_velocity, double utc_seconds, [Out] double[] ecef_velocity);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_ecef_to_eci_velocity([In] double[] ecef_position, [In] double[] ecef_velocity, double utc_seconds, [Out] double[] eci_velocity);

        // === 工具函数 ===
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int j2_compute_gmst(double utc_seconds, out double gmst);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double j2_normalize_angle(double angle);
    }

    /// <summary>
    /// J2轨道传播器的C#封装类
    /// </summary>
    public class J2OrbitPropagator : IDisposable
    {
        private IntPtr handle;
        private bool disposed = false;

        /// <summary>
        /// 初始化J2轨道传播器
        /// </summary>
        /// <param name="initialElements">初始轨道要素</param>
        public J2OrbitPropagator(COrbitalElements initialElements)
        {
            handle = J2OrbitPropagatorNative.j2_propagator_create(ref initialElements);
            if (handle == IntPtr.Zero)
            {
                throw new InvalidOperationException("无法创建J2轨道传播器实例");
            }
        }

        /// <summary>
        /// 析构函数
        /// </summary>
        ~J2OrbitPropagator()
        {
            Dispose(false);
        }

        /// <summary>
        /// 释放资源
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (handle != IntPtr.Zero)
                {
                    J2OrbitPropagatorNative.j2_propagator_destroy(handle);
                    handle = IntPtr.Zero;
                }
                disposed = true;
            }
        }

        private void CheckDisposed()
        {
            if (disposed)
            {
                throw new ObjectDisposedException(nameof(J2OrbitPropagator));
            }
        }

        /// <summary>
        /// 将轨道传播到指定时间
        /// </summary>
        /// <param name="targetTime">目标时间 (s)</param>
        /// <returns>传播后的轨道要素</returns>
        public COrbitalElements Propagate(double targetTime)
        {
            CheckDisposed();
            
            int result = J2OrbitPropagatorNative.j2_propagator_propagate(handle, targetTime, out COrbitalElements elements);
            if (result != 0)
            {
                throw new InvalidOperationException("轨道传播失败");
            }
            
            return elements;
        }

        /// <summary>
        /// 从轨道要素计算状态向量
        /// </summary>
        /// <param name="elements">轨道要素</param>
        /// <returns>状态向量</returns>
        public CStateVector ElementsToState(COrbitalElements elements)
        {
            CheckDisposed();
            
            int result = J2OrbitPropagatorNative.j2_propagator_elements_to_state(handle, ref elements, out CStateVector state);
            if (result != 0)
            {
                throw new InvalidOperationException("轨道要素到状态向量转换失败");
            }
            
            return state;
        }

        /// <summary>
        /// 从状态向量计算轨道要素
        /// </summary>
        /// <param name="state">状态向量</param>
        /// <param name="time">对应的时间 (s)</param>
        /// <returns>轨道要素</returns>
        public COrbitalElements StateToElements(CStateVector state, double time)
        {
            CheckDisposed();
            
            int result = J2OrbitPropagatorNative.j2_propagator_state_to_elements(handle, ref state, time, out COrbitalElements elements);
            if (result != 0)
            {
                throw new InvalidOperationException("状态向量到轨道要素转换失败");
            }
            
            return elements;
        }

        /// <summary>
        /// 设置积分步长
        /// </summary>
        /// <param name="stepSize">步长 (s)</param>
        public void SetStepSize(double stepSize)
        {
            CheckDisposed();
            
            int result = J2OrbitPropagatorNative.j2_propagator_set_step_size(handle, stepSize);
            if (result != 0)
            {
                throw new InvalidOperationException("设置步长失败");
            }
        }

        /// <summary>
        /// 获取当前积分步长
        /// </summary>
        /// <returns>步长 (s)</returns>
        public double GetStepSize()
        {
            CheckDisposed();
            
            int result = J2OrbitPropagatorNative.j2_propagator_get_step_size(handle, out double stepSize);
            if (result != 0)
            {
                throw new InvalidOperationException("获取步长失败");
            }
            
            return stepSize;
        }

        /// <summary>
        /// 启用或禁用自适应步长
        /// </summary>
        /// <param name="enable">是否启用</param>
        public void SetAdaptiveStepSize(bool enable)
        {
            CheckDisposed();
            
            int result = J2OrbitPropagatorNative.j2_propagator_set_adaptive_step_size(handle, enable ? 1 : 0);
            if (result != 0)
            {
                throw new InvalidOperationException("设置自适应步长失败");
            }
        }

        /// <summary>
        /// 设置自适应步长参数
        /// </summary>
        /// <param name="tolerance">误差容忍度</param>
        /// <param name="minStep">最小步长 (s)</param>
        /// <param name="maxStep">最大步长 (s)</param>
        public void SetAdaptiveParameters(double tolerance, double minStep, double maxStep)
        {
            CheckDisposed();
            
            int result = J2OrbitPropagatorNative.j2_propagator_set_adaptive_parameters(handle, tolerance, minStep, maxStep);
            if (result != 0)
            {
                throw new InvalidOperationException("设置自适应步长参数失败");
            }
        }
    }

    /// <summary>
    /// 坐标转换和工具函数的静态类
    /// </summary>
    public static class J2Utils
    {
        /// <summary>
        /// ECI到ECEF坐标转换
        /// </summary>
        /// <param name="eciPosition">ECI位置向量</param>
        /// <param name="utcSeconds">UTC时间 (秒)</param>
        /// <returns>ECEF位置向量</returns>
        public static double[] EciToEcefPosition(double[] eciPosition, double utcSeconds)
        {
            if (eciPosition.Length != 3)
                throw new ArgumentException("位置向量必须包含3个元素");
            
            double[] ecefPosition = new double[3];
            int result = J2OrbitPropagatorNative.j2_eci_to_ecef_position(eciPosition, utcSeconds, ecefPosition);
            if (result != 0)
            {
                throw new InvalidOperationException("ECI到ECEF坐标转换失败");
            }
            
            return ecefPosition;
        }

        /// <summary>
        /// ECEF到ECI坐标转换
        /// </summary>
        /// <param name="ecefPosition">ECEF位置向量</param>
        /// <param name="utcSeconds">UTC时间 (秒)</param>
        /// <returns>ECI位置向量</returns>
        public static double[] EcefToEciPosition(double[] ecefPosition, double utcSeconds)
        {
            if (ecefPosition.Length != 3)
                throw new ArgumentException("位置向量必须包含3个元素");
            
            double[] eciPosition = new double[3];
            int result = J2OrbitPropagatorNative.j2_ecef_to_eci_position(ecefPosition, utcSeconds, eciPosition);
            if (result != 0)
            {
                throw new InvalidOperationException("ECEF到ECI坐标转换失败");
            }
            
            return eciPosition;
        }

        /// <summary>
        /// ECI到ECEF速度转换
        /// </summary>
        /// <param name="eciPosition">ECI位置向量</param>
        /// <param name="eciVelocity">ECI速度向量</param>
        /// <param name="utcSeconds">UTC时间 (秒)</param>
        /// <returns>ECEF速度向量</returns>
        public static double[] EciToEcefVelocity(double[] eciPosition, double[] eciVelocity, double utcSeconds)
        {
            if (eciPosition.Length != 3 || eciVelocity.Length != 3)
                throw new ArgumentException("位置和速度向量必须包含3个元素");
            
            double[] ecefVelocity = new double[3];
            int result = J2OrbitPropagatorNative.j2_eci_to_ecef_velocity(eciPosition, eciVelocity, utcSeconds, ecefVelocity);
            if (result != 0)
            {
                throw new InvalidOperationException("ECI到ECEF速度转换失败");
            }
            
            return ecefVelocity;
        }

        /// <summary>
        /// ECEF到ECI速度转换
        /// </summary>
        /// <param name="ecefPosition">ECEF位置向量</param>
        /// <param name="ecefVelocity">ECEF速度向量</param>
        /// <param name="utcSeconds">UTC时间 (秒)</param>
        /// <returns>ECI速度向量</returns>
        public static double[] EcefToEciVelocity(double[] ecefPosition, double[] ecefVelocity, double utcSeconds)
        {
            if (ecefPosition.Length != 3 || ecefVelocity.Length != 3)
                throw new ArgumentException("位置和速度向量必须包含3个元素");
            
            double[] eciVelocity = new double[3];
            int result = J2OrbitPropagatorNative.j2_ecef_to_eci_velocity(ecefPosition, ecefVelocity, utcSeconds, eciVelocity);
            if (result != 0)
            {
                throw new InvalidOperationException("ECEF到ECI速度转换失败");
            }
            
            return eciVelocity;
        }

        /// <summary>
        /// 计算格林威治平恒星时
        /// </summary>
        /// <param name="utcSeconds">UTC时间 (秒)</param>
        /// <returns>GMST (弧度)</returns>
        public static double ComputeGmst(double utcSeconds)
        {
            int result = J2OrbitPropagatorNative.j2_compute_gmst(utcSeconds, out double gmst);
            if (result != 0)
            {
                throw new InvalidOperationException("计算GMST失败");
            }
            
            return gmst;
        }

        /// <summary>
        /// 角度归一化到[0, 2π)范围
        /// </summary>
        /// <param name="angle">输入角度 (弧度)</param>
        /// <returns>归一化后的角度 (弧度)</returns>
        public static double NormalizeAngle(double angle)
        {
            return J2OrbitPropagatorNative.j2_normalize_angle(angle);
        }
    }

    /// <summary>
    /// 示例程序
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("J2轨道传播器C#绑定示例");
            Console.WriteLine(new string('=', 40));

            try
            {
                // 定义初始轨道要素（ISS轨道参数示例）
                var initialElements = new COrbitalElements
                {
                    a = 6.78e6,        // 半长轴 (m)
                    e = 0.0001,        // 偏心率
                    i = 0.9006,        // 倾角 (rad) ≈ 51.6°
                    O = 0.0,           // 升交点赤经 (rad)
                    w = 0.0,           // 近地点幅角 (rad)
                    M = 0.0,           // 平近点角 (rad)
                    t = 0.0            // 历元时间 (s)
                };

                Console.WriteLine($"初始轨道要素: {initialElements}");

                // 创建传播器实例
                using (var propagator = new J2OrbitPropagator(initialElements))
                {
                    // 设置积分步长
                    propagator.SetStepSize(60.0);  // 60秒
                    Console.WriteLine($"积分步长: {propagator.GetStepSize()} 秒");

                    // 传播轨道
                    double targetTime = 3600.0;  // 1小时后
                    var propagatedElements = propagator.Propagate(targetTime);
                    Console.WriteLine($"\n传播后轨道要素 (t={targetTime}s):");
                    Console.WriteLine($"  {propagatedElements}");

                    // 轨道要素到状态向量转换
                    var state = propagator.ElementsToState(propagatedElements);
                    Console.WriteLine($"\n状态向量:");
                    Console.WriteLine($"  位置 (m): [{state.r[0]:F3}, {state.r[1]:F3}, {state.r[2]:F3}]");
                    Console.WriteLine($"  速度 (m/s): [{state.v[0]:F3}, {state.v[1]:F3}, {state.v[2]:F3}]");

                    // 状态向量到轨道要素转换（验证）
                    var recoveredElements = propagator.StateToElements(state, targetTime);
                    Console.WriteLine($"\n恢复的轨道要素:");
                    Console.WriteLine($"  {recoveredElements}");

                    // 坐标转换示例
                    Console.WriteLine($"\n坐标转换示例:");
                    double[] eciPos = state.r;
                    double utcTime = targetTime;

                    // ECI到ECEF转换
                    double[] ecefPos = J2Utils.EciToEcefPosition(eciPos, utcTime);
                    Console.WriteLine($"  ECI位置: [{eciPos[0]:F3}, {eciPos[1]:F3}, {eciPos[2]:F3}] m");
                    Console.WriteLine($"  ECEF位置: [{ecefPos[0]:F3}, {ecefPos[1]:F3}, {ecefPos[2]:F3}] m");

                    // ECEF到ECI转换（验证）
                    double[] recoveredEci = J2Utils.EcefToEciPosition(ecefPos, utcTime);
                    Console.WriteLine($"  恢复ECI位置: [{recoveredEci[0]:F3}, {recoveredEci[1]:F3}, {recoveredEci[2]:F3}] m");

                    // 计算GMST
                    double gmst = J2Utils.ComputeGmst(utcTime);
                    Console.WriteLine($"  GMST: {gmst:F6} rad ({gmst * 180 / Math.PI:F2}°)");

                    // 角度归一化示例
                    double testAngle = 7.5;  // > 2π
                    double normalized = J2Utils.NormalizeAngle(testAngle);
                    Console.WriteLine($"  角度归一化: {testAngle:F3} → {normalized:F3} rad");

                    Console.WriteLine("\n示例运行成功！");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"错误: {ex.Message}");
                Environment.Exit(1);
            }

            Console.WriteLine("\n按任意键退出...");
            Console.ReadKey();
        }
    }
}