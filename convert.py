#!/usr/bin/env python3
"""
VideoJAM分布式检查点转Safetensors - 无依赖版本
"""

import os
import argparse
import torch
import glob
import json
from collections import OrderedDict
from safetensors.torch import save_file

class SimpleCheckpointConverter:
    def __init__(self):
        self.supported_formats = ['.distcp', '.bin', '.pth', '.pt']
    
    def detect_distcp_files(self, directory):
        """检测分布式检查点文件"""
        distcp_files = glob.glob(os.path.join(directory, "*.distcp"))
        bin_files = glob.glob(os.path.join(directory, "*.bin"))
        return distcp_files + bin_files
    
    def load_shard_file(self, file_path):
        """加载单个分片文件"""
        print(f"加载分片: {os.path.basename(file_path)}")
        
        try:
            # 尝试作为PyTorch文件加载
            return torch.load(file_path, map_location='cpu')
        except:
            # 如果失败，尝试其他方法
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                print(f"  文件大小: {len(data)} 字节")
                # 这里可以添加自定义解析逻辑
                return None
            except Exception as e:
                print(f"  加载失败: {e}")
                return None
    
    def convert_distributed_simple(self, source_dir, output_path):
        """简单转换方法 - 不依赖torch.distributed.checkpoint"""
        print(f"开始转换: {source_dir} -> {output_path}")
        
        # 检查源目录
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"目录不存在: {source_dir}")
        
        # 查找所有分片文件
        shard_files = self.detect_distcp_files(source_dir)
        if not shard_files:
            raise ValueError(f"在 {source_dir} 中未找到.distcp或.bin文件")
        
        print(f"找到 {len(shard_files)} 个分片文件")
        
        # 按文件名排序（假设文件名包含rank和shard信息）
        shard_files.sort()
        
        # 合并所有分片
        full_state_dict = OrderedDict()
        successful_shards = 0
        
        for shard_file in shard_files:
            shard_data = self.load_shard_file(shard_file)
            if shard_data and isinstance(shard_data, dict):
                full_state_dict.update(shard_data)
                successful_shards += 1
            else:
                print(f"警告: 无法加载 {shard_file}")
        
        if successful_shards == 0:
            raise ValueError("无法加载任何分片文件")
        
        print(f"成功加载 {successful_shards}/{len(shard_files)} 个分片")
        print(f"合并后的状态字典包含 {len(full_state_dict)} 个键")
        
        # 保存为Safetensors格式
        save_file(dict(full_state_dict), output_path)
        print(f"✓ 成功保存到: {output_path}")
        
        # 生成元数据
        metadata = {
            "source_directory": source_dir,
            "shard_files_processed": successful_shards,
            "total_keys": len(full_state_dict),
            "conversion_method": "simple_manual_merge"
        }
        
        metadata_path = output_path.replace('.safetensors', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True
    
    def validate_conversion(self, safetensors_path):
        """验证转换结果"""
        from safetensors.torch import load_file
        
        try:
            state_dict = load_file(safetensors_path)
            print(f"验证: 文件包含 {len(state_dict)} 个键")
            
            # 显示前几个键作为示例
            sample_keys = list(state_dict.keys())[:5]
            print("示例键:")
            for i, key in enumerate(sample_keys):
                tensor = state_dict[key]
                print(f"  {i+1}. {key}: {tuple(tensor.shape)}")
            
            return True
        except Exception as e:
            print(f"验证失败: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="简单分布式检查点转换工具")
    parser.add_argument("--source", required=True, help="源检查点目录")
    parser.add_argument("--output", required=True, help="输出Safetensors文件路径")
    parser.add_argument("--validate", action="store_true", help="转换后验证结果")
    
    args = parser.parse_args()
    
    converter = SimpleCheckpointConverter()
    
    try:
        # 执行转换
        success = converter.convert_distributed_simple(args.source, args.output)
        
        if success:
            print("🎉 转换成功完成！")
            
            # 可选验证
            if args.validate:
                print("开始验证转换结果...")
                converter.validate_conversion(args.output)
        else:
            print("❌ 转换失败")
            return 1
            
    except Exception as e:
        print(f"❌ 转换过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())