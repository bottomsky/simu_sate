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

        static void Main()
        {
            // 验证C#结构体与C结构体的大小一致
            AssertSize<COrbitalElements>(sizeof(double) * 7);
            AssertSize<CStateVector>(sizeof(double) * 3 * 2);

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

            Console.WriteLine("Memory layout and roundtrip tests passed.");
        }
    }
}