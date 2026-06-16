#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 data-1005 目录中的所有 .doc 文件转换为 .txt 文件
在同级目录下生成同名的 txt 文件
"""

import os
import sys
from pathlib import Path



def extract_text_with_antiword(doc_path):
    """使用 antiword 命令行工具提取文本（仅支持 .doc）"""
    try:
        import subprocess
        result = subprocess.run(['antiword', doc_path], 
                              capture_output=True, 
                              text=True, 
                              timeout=30)
        if result.returncode == 0:
            return result.stdout
        else:
            return None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"  antiword 提取失败: {e}")
        return None


def extract_text_with_python_docx(doc_path):
    """使用 python-docx 提取文本（仅支持 .docx）"""
    try:
        from docx import Document
        doc = Document(doc_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return '\n'.join(text)
    except Exception as e:
        print(f"  python-docx 提取失败: {e}")
        return None


def extract_text_with_doc2txt(doc_path):
    """使用 doc2txt 提取文本（支持 .doc）"""
    try:
        import subprocess
        # 尝试使用 catdoc
        result = subprocess.run(['catdoc', doc_path], 
                              capture_output=True, 
                              text=True, 
                              timeout=30,
                              encoding='utf-8',
                              errors='ignore')
        if result.returncode == 0:
            return result.stdout
        else:
            return None
    except FileNotFoundError:
        return None
    except Exception as e:
        return None


def extract_text_with_libreoffice(doc_path):
    """使用 LibreOffice 转换（支持 .doc 和 .docx）"""
    try:
        import subprocess
        import tempfile
        
        # 使用临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            # 转换为 txt
            result = subprocess.run([
                'libreoffice', '--headless', '--convert-to', 'txt:Text',
                '--outdir', tmpdir, doc_path
            ], capture_output=True, timeout=30)
            
            if result.returncode == 0:
                # 读取生成的 txt 文件
                txt_filename = Path(doc_path).stem + '.txt'
                txt_path = os.path.join(tmpdir, txt_filename)
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                        return f.read()
        return None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"  LibreOffice 提取失败: {e}")
        return None


def extract_doc_text(doc_path):
    """
    尝试多种方法提取 doc 文件中的文本
    优先级：textract > antiword > catdoc > LibreOffice > python-docx
    """
    print(f"正在处理: {doc_path}")
    
    
    # 方法4: LibreOffice
    text = extract_text_with_libreoffice(doc_path)
    if text:
        print(f"  ✓ 使用 LibreOffice 提取成功")
        return text
    
    # 方法5: python-docx (适用于 .docx)
    if doc_path.lower().endswith('.doc'):
        text = extract_text_with_python_docx(doc_path)
        if text:
            print(f"  ✓ 使用 python-docx 提取成功")
            return text
    
    print(f"  ✗ 所有方法都无法提取文本")
    return None


def process_directory(root_dir):
    """递归处理目录中的所有 .doc 和 .docx 文件"""
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"错误: 目录不存在: {root_dir}")
        return
    
    # 查找所有 .doc 和 .docx 文件
    doc_files = list(root_path.rglob('*.doc'))
    doc_files.extend(root_path.rglob('*.docx'))
    
    if not doc_files:
        print(f"未找到任何 .doc 或 .docx 文件")
        return
    
    print(f"找到 {len(doc_files)} 个文件")
    print("=" * 80)
    
    success_count = 0
    failed_count = 0
    
    for doc_file in doc_files:
        try:
            # 提取文本
            text = extract_doc_text(str(doc_file))
            
            if text:
                # 生成 txt 文件路径（与 doc 文件同名，同目录）
                txt_file = doc_file.with_suffix('.txt')
                
                # 写入文本文件
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"  ✓ 已生成: {txt_file}")
                success_count += 1
            else:
                print(f"  ✗ 提取失败，跳过", doc_file)
                failed_count += 1
                break
                
        except Exception as e:
            print(f"  ✗ 处理出错: {e}")
            failed_count += 1
            break
        
        print("-" * 80)
    
    print(f"\n处理完成!")
    print(f"成功: {success_count} 个")
    print(f"失败: {failed_count} 个")


def print_installation_guide():
    """打印安装指南"""
    print("\n" + "=" * 80)
    print("如果脚本无法运行，请先安装必要的依赖：")
    print("=" * 80)
    print("\n方案1（推荐）：使用 textract")
    print("  pip install textract")
    print("  # 还需要安装系统依赖，参考: https://textract.readthedocs.io")
    print("\n方案2：使用 antiword（仅限 .doc）")
    print("  Ubuntu/Debian: sudo apt-get install antiword")
    print("  CentOS/RHEL: sudo yum install antiword")
    print("  macOS: brew install antiword")
    print("\n方案3：使用 catdoc（仅限 .doc）")
    print("  Ubuntu/Debian: sudo apt-get install catdoc")
    print("  CentOS/RHEL: sudo yum install catdoc")
    print("  macOS: brew install catdoc")
    print("\n方案4：使用 LibreOffice")
    print("  Ubuntu/Debian: sudo apt-get install libreoffice")
    print("  CentOS/RHEL: sudo yum install libreoffice")
    print("  macOS: brew install --cask libreoffice")
    print("\n方案5：使用 python-docx（仅限 .docx）")
    print("  pip install python-docx")
    print("=" * 80 + "\n")


def help():
    print_installation_guide()


def doc2txt(data_dir):
    process_directory(data_dir)

def main():
    # 数据目录
    data_dir = '/media/codingma/LLM/data-1005'
    
    # 打印安装指南
    print_installation_guide()
    
    # 处理目录
    process_directory(data_dir)


if __name__ == '__main__':
    main()

