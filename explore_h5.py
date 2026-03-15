# -*- coding: UTF-8 -*-
import h5py
import numpy as np
import argparse

def explore_h5_structure(h5_file):
    """
    探索H5文件的结构
    """
    print(f"探索H5文件: {h5_file}")
    print("=" * 50)
    
    with h5py.File(h5_file, 'r') as f:
        def print_structure(name, obj):
            print(f"{name}: {type(obj).__name__}")
            if isinstance(obj, h5py.Dataset):
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                if obj.shape and len(obj.shape) <= 2 and np.prod(obj.shape) <= 20:
                    print(f"  Data: {obj[:]}")
            elif isinstance(obj, h5py.Group):
                print(f"  Keys: {list(obj.keys())}")
        
        f.visititems(print_structure)
        
        print("\n" + "=" * 50)
        print("详细数据集信息:")
        print("=" * 50)
        
        def print_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"\n数据集: {name}")
                print(f"  形状: {obj.shape}")
                print(f"  数据类型: {obj.dtype}")
                print(f"  属性: {dict(obj.attrs)}")
                
                # 如果数据集不太大，显示一些样本数据
                if obj.shape and np.prod(obj.shape) <= 100:
                    print(f"  数据样本: {obj[:]}")
                elif obj.shape:
                    print(f"  数据样本 (前5个): {obj.flat[:5]}")
        
        f.visititems(print_datasets)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='探索H5文件结构')
    parser.add_argument('--h5_file', type=str, required=True, help='H5文件路径')
    args = parser.parse_args()
    
    explore_h5_structure(args.h5_file)
