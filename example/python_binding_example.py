#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J2轨道传播器Python绑定示例

该示例展示了如何通过ctypes调用J2轨道传播器的C接口动态库。
包括轨道传播、坐标转换等功能的使用方法。

使用前请确保已编译生成动态库文件：
- Windows: j2_orbit_propagator.dll
- Linux: libj2_orbit_propagator.so
- macOS: libj2_orbit_propagator.dylib
"""

import ctypes
import platform
import os
from ctypes import Structure, c_double, c_void_p, c_int, POINTER

# 根据操作系统确定动态库文件名
system = platform.system()
if system == "Windows":
    lib_name = "j2_orbit_propagator.dll"
elif system == "Darwin":
    lib_name = "libj2_orbit_propagator.dylib"
else:
    lib_name = "libj2_orbit_propagator.so"

# 加载动态库
try:
    # 首先尝试从当前目录加载
    current_dir = os.path.dirname(__file__)
    lib_path = os.path.join(current_dir, lib_name)
    if not os.path.exists(lib_path):
        # 如果不存在，尝试从build目录加载
        lib_path = os.path.join(current_dir, "..", "build", "Release", lib_name)
        if not os.path.exists(lib_path):
            # 最后尝试系统路径
            lib_path = lib_name
    
    j2_lib = ctypes.CDLL(lib_path)
except OSError as e:
    print(f"无法加载动态库 {lib_name}: {e}")
    print("请确保已编译生成动态库文件")
    exit(1)

# 定义C结构体
class COrbitalElements(Structure):
    """C格式的轨道要素结构体"""
    _fields_ = [
        ("a", c_double),    # 半长轴 (m)
        ("e", c_double),    # 偏心率
        ("i", c_double),    # 倾角 (rad)
        ("O", c_double),    # 升交点赤经 (rad)
        ("w", c_double),    # 近地点幅角 (rad)
        ("M", c_double),    # 平近点角 (rad)
        ("t", c_double),    # 历元时间 (s)
    ]
    
    def __str__(self):
        return (f"COrbitalElements(a={self.a:.3f}, e={self.e:.6f}, "
                f"i={self.i:.6f}, O={self.O:.6f}, w={self.w:.6f}, "
                f"M={self.M:.6f}, t={self.t:.3f})")

class CStateVector(Structure):
    """C格式的状态向量结构体"""
    _fields_ = [
        ("r", c_double * 3),  # 位置矢量 (m)
        ("v", c_double * 3),  # 速度矢量 (m/s)
    ]
    
    def __str__(self):
        return (f"CStateVector(r=[{self.r[0]:.3f}, {self.r[1]:.3f}, {self.r[2]:.3f}], "
                f"v=[{self.v[0]:.3f}, {self.v[1]:.3f}, {self.v[2]:.3f}])")

# 定义函数原型
# 创建和销毁函数
j2_lib.j2_propagator_create.argtypes = [POINTER(COrbitalElements)]
j2_lib.j2_propagator_create.restype = c_void_p

j2_lib.j2_propagator_destroy.argtypes = [c_void_p]
j2_lib.j2_propagator_destroy.restype = None

# 轨道传播函数
j2_lib.j2_propagator_propagate.argtypes = [c_void_p, c_double, POINTER(COrbitalElements)]
j2_lib.j2_propagator_propagate.restype = c_int

j2_lib.j2_propagator_elements_to_state.argtypes = [c_void_p, POINTER(COrbitalElements), POINTER(CStateVector)]
j2_lib.j2_propagator_elements_to_state.restype = c_int

j2_lib.j2_propagator_state_to_elements.argtypes = [c_void_p, POINTER(CStateVector), c_double, POINTER(COrbitalElements)]
j2_lib.j2_propagator_state_to_elements.restype = c_int

# 参数设置函数
j2_lib.j2_propagator_set_step_size.argtypes = [c_void_p, c_double]
j2_lib.j2_propagator_set_step_size.restype = c_int

j2_lib.j2_propagator_get_step_size.argtypes = [c_void_p, POINTER(c_double)]
j2_lib.j2_propagator_get_step_size.restype = c_int

j2_lib.j2_propagator_set_adaptive_step_size.argtypes = [c_void_p, c_int]
j2_lib.j2_propagator_set_adaptive_step_size.restype = c_int

# 坐标转换函数
j2_lib.j2_eci_to_ecef_position.argtypes = [POINTER(c_double), c_double, POINTER(c_double)]
j2_lib.j2_eci_to_ecef_position.restype = c_int

j2_lib.j2_ecef_to_eci_position.argtypes = [POINTER(c_double), c_double, POINTER(c_double)]
j2_lib.j2_ecef_to_eci_position.restype = c_int

# 工具函数
j2_lib.j2_compute_gmst.argtypes = [c_double, POINTER(c_double)]
j2_lib.j2_compute_gmst.restype = c_int

j2_lib.j2_normalize_angle.argtypes = [c_double]
j2_lib.j2_normalize_angle.restype = c_double

class J2OrbitPropagator:
    """J2轨道传播器Python封装类"""
    
    def __init__(self, initial_elements):
        """
        初始化J2轨道传播器
        
        Args:
            initial_elements: 初始轨道要素字典，包含键：
                - a: 半长轴 (m)
                - e: 偏心率
                - i: 倾角 (rad)
                - O: 升交点赤经 (rad)
                - w: 近地点幅角 (rad)
                - M: 平近点角 (rad)
                - t: 历元时间 (s)
        """
        self.c_elements = COrbitalElements(
            a=initial_elements['a'],
            e=initial_elements['e'],
            i=initial_elements['i'],
            O=initial_elements['O'],
            w=initial_elements['w'],
            M=initial_elements['M'],
            t=initial_elements['t']
        )
        
        self.handle = j2_lib.j2_propagator_create(ctypes.byref(self.c_elements))
        if not self.handle:
            raise RuntimeError("无法创建J2轨道传播器实例")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'handle') and self.handle:
            j2_lib.j2_propagator_destroy(self.handle)
    
    def propagate(self, target_time):
        """
        将轨道传播到指定时间
        
        Args:
            target_time: 目标时间 (s)
            
        Returns:
            dict: 传播后的轨道要素
        """
        result = COrbitalElements()
        ret = j2_lib.j2_propagator_propagate(self.handle, target_time, ctypes.byref(result))
        if ret != 0:
            raise RuntimeError("轨道传播失败")
        
        return {
            'a': result.a,
            'e': result.e,
            'i': result.i,
            'O': result.O,
            'w': result.w,
            'M': result.M,
            't': result.t
        }
    
    def elements_to_state(self, elements):
        """
        从轨道要素计算状态向量
        
        Args:
            elements: 轨道要素字典
            
        Returns:
            dict: 状态向量，包含'r'(位置)和'v'(速度)
        """
        c_elements = COrbitalElements(
            a=elements['a'], e=elements['e'], i=elements['i'],
            O=elements['O'], w=elements['w'], M=elements['M'], t=elements['t']
        )
        
        state = CStateVector()
        ret = j2_lib.j2_propagator_elements_to_state(self.handle, ctypes.byref(c_elements), ctypes.byref(state))
        if ret != 0:
            raise RuntimeError("轨道要素到状态向量转换失败")
        
        return {
            'r': [state.r[0], state.r[1], state.r[2]],
            'v': [state.v[0], state.v[1], state.v[2]]
        }
    
    def state_to_elements(self, state, time):
        """
        从状态向量计算轨道要素
        
        Args:
            state: 状态向量字典，包含'r'(位置)和'v'(速度)
            time: 对应的时间 (s)
            
        Returns:
            dict: 轨道要素
        """
        c_state = CStateVector()
        c_state.r[0], c_state.r[1], c_state.r[2] = state['r']
        c_state.v[0], c_state.v[1], c_state.v[2] = state['v']
        
        elements = COrbitalElements()
        ret = j2_lib.j2_propagator_state_to_elements(self.handle, ctypes.byref(c_state), time, ctypes.byref(elements))
        if ret != 0:
            raise RuntimeError("状态向量到轨道要素转换失败")
        
        return {
            'a': elements.a,
            'e': elements.e,
            'i': elements.i,
            'O': elements.O,
            'w': elements.w,
            'M': elements.M,
            't': elements.t
        }
    
    def set_step_size(self, step_size):
        """设置积分步长"""
        ret = j2_lib.j2_propagator_set_step_size(self.handle, step_size)
        if ret != 0:
            raise RuntimeError("设置步长失败")
    
    def get_step_size(self):
        """获取当前积分步长"""
        step_size = c_double()
        ret = j2_lib.j2_propagator_get_step_size(self.handle, ctypes.byref(step_size))
        if ret != 0:
            raise RuntimeError("获取步长失败")
        return step_size.value
    
    def set_adaptive_step_size(self, enable):
        """启用或禁用自适应步长"""
        ret = j2_lib.j2_propagator_set_adaptive_step_size(self.handle, 1 if enable else 0)
        if ret != 0:
            raise RuntimeError("设置自适应步长失败")

def eci_to_ecef_position(eci_position, utc_seconds):
    """ECI到ECEF坐标转换"""
    eci_array = (c_double * 3)(*eci_position)
    ecef_array = (c_double * 3)()
    
    ret = j2_lib.j2_eci_to_ecef_position(eci_array, utc_seconds, ecef_array)
    if ret != 0:
        raise RuntimeError("ECI到ECEF坐标转换失败")
    
    return [ecef_array[0], ecef_array[1], ecef_array[2]]

def ecef_to_eci_position(ecef_position, utc_seconds):
    """ECEF到ECI坐标转换"""
    ecef_array = (c_double * 3)(*ecef_position)
    eci_array = (c_double * 3)()
    
    ret = j2_lib.j2_ecef_to_eci_position(ecef_array, utc_seconds, eci_array)
    if ret != 0:
        raise RuntimeError("ECEF到ECI坐标转换失败")
    
    return [eci_array[0], eci_array[1], eci_array[2]]

def compute_gmst(utc_seconds):
    """计算格林威治平恒星时"""
    gmst = c_double()
    ret = j2_lib.j2_compute_gmst(utc_seconds, ctypes.byref(gmst))
    if ret != 0:
        raise RuntimeError("计算GMST失败")
    return gmst.value

def normalize_angle(angle):
    """角度归一化"""
    return j2_lib.j2_normalize_angle(angle)

def main():
    """示例主函数"""
    print("J2轨道传播器Python绑定示例")
    print("=" * 40)
    
    # 定义初始轨道要素（ISS轨道参数示例）
    initial_elements = {
        'a': 6.78e6,        # 半长轴 (m)
        'e': 0.0001,        # 偏心率
        'i': 0.9006,        # 倾角 (rad) ≈ 51.6°
        'O': 0.0,           # 升交点赤经 (rad)
        'w': 0.0,           # 近地点幅角 (rad)
        'M': 0.0,           # 平近点角 (rad)
        't': 0.0            # 历元时间 (s)
    }
    
    try:
        # 创建传播器实例
        propagator = J2OrbitPropagator(initial_elements)
        print(f"初始轨道要素: {initial_elements}")
        
        # 设置积分步长
        propagator.set_step_size(60.0)  # 60秒
        print(f"积分步长: {propagator.get_step_size()} 秒")
        
        # 传播轨道
        target_time = 3600.0  # 1小时后
        propagated_elements = propagator.propagate(target_time)
        print(f"\n传播后轨道要素 (t={target_time}s):")
        for key, value in propagated_elements.items():
            print(f"  {key}: {value:.6f}")
        
        # 轨道要素到状态向量转换
        state = propagator.elements_to_state(propagated_elements)
        print(f"\n状态向量:")
        print(f"  位置 (m): [{state['r'][0]:.3f}, {state['r'][1]:.3f}, {state['r'][2]:.3f}]")
        print(f"  速度 (m/s): [{state['v'][0]:.3f}, {state['v'][1]:.3f}, {state['v'][2]:.3f}]")
        
        # 状态向量到轨道要素转换（验证）
        recovered_elements = propagator.state_to_elements(state, target_time)
        print(f"\n恢复的轨道要素:")
        for key, value in recovered_elements.items():
            print(f"  {key}: {value:.6f}")
        
        # 坐标转换示例
        print(f"\n坐标转换示例:")
        eci_pos = state['r']
        utc_time = target_time
        
        # ECI到ECEF转换
        ecef_pos = eci_to_ecef_position(eci_pos, utc_time)
        print(f"  ECI位置: [{eci_pos[0]:.3f}, {eci_pos[1]:.3f}, {eci_pos[2]:.3f}] m")
        print(f"  ECEF位置: [{ecef_pos[0]:.3f}, {ecef_pos[1]:.3f}, {ecef_pos[2]:.3f}] m")
        
        # ECEF到ECI转换（验证）
        recovered_eci = ecef_to_eci_position(ecef_pos, utc_time)
        print(f"  恢复ECI位置: [{recovered_eci[0]:.3f}, {recovered_eci[1]:.3f}, {recovered_eci[2]:.3f}] m")
        
        # 计算GMST
        gmst = compute_gmst(utc_time)
        print(f"  GMST: {gmst:.6f} rad ({gmst * 180 / 3.14159:.2f}°)")
        
        # 角度归一化示例
        test_angle = 7.5  # > 2π
        normalized = normalize_angle(test_angle)
        print(f"  角度归一化: {test_angle:.3f} → {normalized:.3f} rad")
        
        print("\n示例运行成功！")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())