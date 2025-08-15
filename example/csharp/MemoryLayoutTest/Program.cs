using System;
using System.Linq;
using System.Runtime.InteropServices;
using J2.Propagator;

namespace MemoryLayoutTest
{
    internal static class Program
    {
        [StructLayout(LayoutKind.Sequential)]
        struct CElems
        {
            public double a, e, i, O, w, M, t;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct CState
        {
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)] public double[] r;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)] public double[] v;
        }

        static void AssertSize<T>(int expected)
        {
            int size = Marshal.SizeOf<T>();
            if (size != expected)
                throw new Exception($"Struct {typeof(T).Name} size mismatch: expected {expected}, actual {size}");
        }

        /// <summary>
        /// 验证指定字段的偏移量，并打印调试信息
        /// </summary>
        static void AssertOffset<T>(string fieldName, int expectedOffset)
        {
            int actualOffset = Marshal.OffsetOf<T>(fieldName).ToInt32();
            Console.WriteLine($"  {typeof(T).Name}.{fieldName}: offset={actualOffset}, expected={expectedOffset}");
            if (actualOffset != expectedOffset)
                throw new Exception($"Field {typeof(T).Name}.{fieldName} offset mismatch: expected {expectedOffset}, actual {actualOffset}");
        }

        /// <summary>
        /// 打印结构体的详细内存布局信息
        /// </summary>
        static void PrintStructLayout<T>(string structName, params (string fieldName, int expectedOffset)[] fields)
        {
            int size = Marshal.SizeOf<T>();
            Console.WriteLine($"{structName} memory layout:");
            Console.WriteLine($"  Total size: {size} bytes");
            
            foreach (var (fieldName, expectedOffset) in fields)
            {
                AssertOffset<T>(fieldName, expectedOffset);
            }
            Console.WriteLine();
        }

        static void TestJ2Orbit()
        {
            Console.WriteLine("Testing J2Orbit functionality...");

            // 验证C#结构体与C结构体的大小一致
            AssertSize<COrbitalElements>(sizeof(double) * 7);
            AssertSize<CStateVector>(sizeof(double) * 3 * 2);

            // 打印与断言偏移（假设Pack=8下double顺序紧凑排列）
            PrintStructLayout<COrbitalElements>(nameof(COrbitalElements),
                (nameof(COrbitalElements.a), 0),
                (nameof(COrbitalElements.e), 8),
                (nameof(COrbitalElements.i), 16),
                (nameof(COrbitalElements.O), 24),
                (nameof(COrbitalElements.w), 32),
                (nameof(COrbitalElements.M), 40),
                (nameof(COrbitalElements.t), 48)
            );

            PrintStructLayout<CStateVector>(nameof(CStateVector),
                (nameof(CStateVector.r), 0),
                (nameof(CStateVector.v), 24) // 3*8 bytes after r
            );

            // 简单端到端：创建传播器，做一次往返转换
            var elems = new COrbitalElements
            {
                a = 7000e3,
                e = 0.001,
                i = 98 * Math.PI / 180.0,
                O = 40 * Math.PI / 180.0,
                w = 20 * Math.PI / 180.0,
                M = 0.5,
                t = 0
            };

            using var propagator = new J2Orbit(elems);

            var state = propagator.ElementsToState(elems);
            var roundtrip = propagator.StateToElements(state, elems.t);

            // 误差检查
            static void CheckClose(string name, double a, double b, double tol)
            {
                if (Math.Abs(a - b) > tol)
                    throw new Exception($"{name} mismatch: {a} vs {b}");
            }

            CheckClose("a", elems.a, roundtrip.a, 1e-3);
            CheckClose("e", elems.e, roundtrip.e, 1e-9);
            CheckClose("i", elems.i, roundtrip.i, 1e-9);
            CheckClose("O", elems.O, roundtrip.O, 1e-9);
            CheckClose("w", elems.w, roundtrip.w, 1e-9);

            Console.WriteLine("J2Orbit memory layout and roundtrip tests passed.");
        }

        static void TestConstellationPropagator()
        {
            Console.WriteLine("Testing ConstellationPropagator functionality...");

            // 验证新结构体的大小
            AssertSize<CCompactOrbitalElements>(sizeof(double) * 6);

            // 偏移断言：6个double紧凑排列
            PrintStructLayout<CCompactOrbitalElements>(nameof(CCompactOrbitalElements),
                (nameof(CCompactOrbitalElements.a), 0),
                (nameof(CCompactOrbitalElements.e), 8),
                (nameof(CCompactOrbitalElements.i), 16),
                (nameof(CCompactOrbitalElements.O), 24),
                (nameof(CCompactOrbitalElements.w), 32),
                (nameof(CCompactOrbitalElements.M), 40)
            );

            // 检查 CUDA 可用性
            bool cudaAvailable = ConstellationPropagator.IsCudaAvailable();
            Console.WriteLine($"CUDA Available: {cudaAvailable}");

            // 创建星座传播器
            var epochTime = 0.0; // J2000.0
            using var constellation = new ConstellationPropagator(epochTime);

            // 添加几个测试卫星
            var satellites = new[]
            {
                new CCompactOrbitalElements
                {
                    a = 7000e3,   // 轨道半长轴 7000 km
                    e = 0.001,    // 偏心率
                    i = 98.0 * Math.PI / 180.0,  // 倾斜角 98 度（太阳同步轨道）
                    O = 0.0,      // 升交点赤经
                    w = 0.0,      // 近地点幅角
                    M = 0.0       // 平近点角
                },
                new CCompactOrbitalElements
                {
                    a = 7000e3,
                    e = 0.001,
                    i = 98.0 * Math.PI / 180.0,
                    O = 60.0 * Math.PI / 180.0,  // 不同升交点赤经
                    w = 0.0,
                    M = 0.0
                },
                new CCompactOrbitalElements
                {
                    a = 7000e3,
                    e = 0.001,
                    i = 98.0 * Math.PI / 180.0,
                    O = 120.0 * Math.PI / 180.0,
                    w = 0.0,
                    M = 0.0
                }
            };

            // 批量添加卫星
            constellation.AddSatellites(satellites);

            int satCount = constellation.GetSatelliteCount();
            Console.WriteLine($"Added {satCount} satellites to constellation.");

            // 测试传播
            var targetTime = 3600.0; // 1 小时后
            constellation.Propagate(targetTime);

            // 获取传播后的状态
            for (int i = 0; i < satCount; i++)
            {
                var elements = constellation.GetSatelliteElements(i);
                var state = constellation.GetSatelliteState(i);
                
                Console.WriteLine($"Satellite {i}: a={elements.a:F0}m, M={elements.M:F3}rad");
                Console.WriteLine($"  Position: [{state.r[0]:F0}, {state.r[1]:F0}, {state.r[2]:F0}] m");
            }

            // 测试获取所有位置
            var allPositions = constellation.GetAllPositions();
            Console.WriteLine($"All positions array length: {allPositions.Length} (expected: {satCount * 3})");

            // 测试脉冲功能
            var deltaVs = new double[satCount * 3];
            for (int i = 0; i < satCount; i++)
            {
                deltaVs[i * 3 + 0] = 1.0; // x 方向 1 m/s
                deltaVs[i * 3 + 1] = 0.0; // y 方向
                deltaVs[i * 3 + 2] = 0.0; // z 方向
            }

            constellation.ApplyImpulseToConstellation(deltaVs, targetTime);
            Console.WriteLine("Applied impulse to all satellites.");

            // 测试单个卫星添加
            var newSat = new CCompactOrbitalElements
            {
                a = 6800e3,
                e = 0.01,
                i = 45.0 * Math.PI / 180.0,
                O = 180.0 * Math.PI / 180.0,
                w = 90.0 * Math.PI / 180.0,
                M = 180.0 * Math.PI / 180.0
            };
            constellation.AddSatellite(newSat);
            Console.WriteLine($"Added single satellite, new count: {constellation.GetSatelliteCount()}");

            // 测试参数设置
            constellation.SetStepSize(10.0); // 10 秒步长
            constellation.SetComputeMode(ComputeMode.CpuScalar);
            constellation.SetAdaptive(true);
            constellation.SetAdaptiveParameters(1e-8, 1.0, 60.0);
            Console.WriteLine("Set constellation parameters.");

            Console.WriteLine("ConstellationPropagator tests passed.");
        }

        static void Main()
        {
            try
            {
                TestJ2Orbit();
                TestConstellationPropagator();
                Console.WriteLine("All tests passed successfully!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Test failed: {ex.Message}");
                Environment.Exit(1);
            }
        }
    }
}