# 试试conda虚拟环境
环境管理允许用户方便安装不同版本的python环境，并在不同环境之间快速地切换。

## 自动配置
```
# ubuntu&wsl2
$ conda env create -f environment_linux.yml
```

## conda 常用命令
```
# 创建虚拟环境
$ conda create -n wde_v0.1 python=3.11 anaconda

# 查看虚拟环境
$ conda env list 

# 激活虚拟环境
$ conda activate wde_v0.1

# 安装依赖, 只会安装在这个虚拟环境里
# pip install .....

# 停用虚拟环境
$ conda deactivate

# 删除虚拟环境
conda remove -n wde_v0.1 --all
```