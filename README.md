# XPS_UPS

一个包含两个模块的光电子谱小工具：

- **UPS 模块**：SECO/功函数（φ）识别，含 HOMO 拼接图预览与导出
- **XPS 模块**：分峰拟合（Pseudo-Voigt + Linear/Shirley 背景），含 **DL 自动拟合**
  （CNN1Dv2 + lmfit 精修，覆盖 C 1s / N 1s / O 1s / S 2p / F 1s；S 2p 自动应用
  spin-orbit doublet 1.18 eV / 2:1 约束）

## 运行

```bash
python XPS_UPS.py
```

启动后先选择 **UPS** 或 **XPS** 模块。XPS 模块的 "Auto-fit" 按钮调用 DL 引擎。

## 依赖

```bash
pip install -r requirements.txt
```

首次启动会自动安装缺库（含 `torch` ~200 MB CPU wheel 和 `lmfit`）。

## DL 模型下载（必读）

仓库**不携带模型文件**（共 38 MB，二进制资产）。请从 GitHub Release 下载：

1. 打开 <https://github.com/Zeshengliu213/XPS_UPS/releases>
2. 下载最新 release 的 `models.zip`
3. 解压到本目录的 `models/` 文件夹，最终结构：

```
XPS_UPS/
├── XPS_UPS.py
├── autofit_engine.py
└── models/
    ├── c1s_v2.pt
    ├── n1s_v2.pt
    ├── o1s_v2.pt
    ├── s2p_v2.pt
    └── f1s_v2.pt
```

未放置 `models/` 时 GUI 仍可启动并使用经典 Peak Fit；只有 "Auto-fit" 按钮会提示
"未找到对应模型"。

## 模型概览

| Orbital | val MAE μ | n_peaks 匹配 | 备注 |
|---------|-----------|--------------|------|
| C 1s    | 0.34 eV   | 65 %         | 真实 C 1s mae 0.54 eV |
| N 1s    | 0.58 eV   | 77 %         | carbazole / amine 优化 |
| O 1s    | 0.49 eV   | —            | 含金属-O / 有机-O / H₂O |
| S 2p    | 0.61 eV   | 75 %         | spin-orbit doublet 自动 |
| F 1s    | 0.51 eV   | —            | TFSI / CF / LiF 覆盖 |

模型训练代码与合成数据脚本见姊妹仓库 `XPSautofit`（即将公开）。

