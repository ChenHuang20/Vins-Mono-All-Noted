# Vins-Mono-All-Noted


- Forked from VINS-Mono: https://github.com/HKUST-Aerial-Robotics/VINS-Mono  

- 注释以 [Vins-Mono-All-Noted](https://github.com/ManiiXu/VINS-Mono-Learning) 为基础。

- 添加了很多关于初始化、后端优化和回环检测的详细注释。

- 从初始化开始，添加了每一步代码的公式推导详解，公式推导主要参考了崔华坤的详解。

## 符号说明

- T 一般代表4×4的变换矩阵 [R | p]，即位姿。
- 如 T_b_c 代表c系到b系的变换， 或者说c系在b参考系下的旋转和平移。