
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import argparse
import os
import glob
import sys
import re





def generate_score(annotation_result, groundtruth, tokenizer, llm):     

    prompt = (
        f"Input: Please score the annotation result base on groundtruth. Following are groundtruth and annotation result, {groundtruth}, {annotation_result}."
        "Output:"
    )
    
    message = [
        {
            "role": "system", "content": (
                '# TASK # Using the groundtruth as reference answer, please score the given annotation results. You should evaluate annotation result based on criteria including structure, content completeness, and content accuracy, with a total score out of 100 points.'
                "# ATTENTION # Task Details:"
                "- The content to be scored is composed of three components: data, analyses, and results. These components are referred to as 'objects', with each object consisting of multiple sub-objects. Each sub-object contains several key-value pairs."
                "- You need to separately score the three 'objects'—data, analyses, and results—each with a maximum score of 100 points. Each score get from three parts of 'Structure', 'Content completeness accounts' and 'Content accuracy'."
                "- Structure accounts for 10 points: This section should focus solely on verifying the correct JSON structure of each object and proper key-value pair structure in sub-objects. Do not score on the actual content of the key-value pairs."
                "- Content completeness accounts for 40 points: This section should score at the sub-object level. Deduct points for missing any sub-object. Note: Sub-objects in annotation result that are similar but not total identical to groundtruth may still qualify as matches. Thoroughly analyze each groundtruth sub-object's semantic correspondence before determining annotation result equivalency. Extra sub-objects may also incur penalties depending on contextual relevance."
                "- Content accuracy accounts for 50 points: This section evaluates the accuracy of matched sub-object’s key-value pairs. For sub-objects deemed semantically matched in the 'Content Completeness' section, deductions are applied based on discrepancies in key-value pair semantics. Again, you must account for potential differences in wording while semantic equivalence. You must prioritize semantic alignment over literal matching."
                "# ATTENTION # For your each scoring process:"
                '- data_id or analysis_id are only unique identifiers for sub-objects. When the same sub-objects are ordered differently, their IDs may vary. Therefore, scoring should focus on the sub-objects content, rather than using IDs to assess consistency, Do not deduct to different ID with same semantical content.'
                "- You need to continuously reflect during the scoring process to ensure that the final score you give is accurate and reasonable. Before giving the final score, you can make reasonable adjustments to your previous scoring process."
                "- You need to provide a complete demonstration of the scoring process, and specify where points were deducted"
                "- Use following format, display the final score results at the end of your response"
                ''' 
                "Final Scores": {
                    "Data":
                    "Analyses":
                    "Results":
                }
                '''
           )
        },

        {"role": "user", "content": prompt},
    ]
    


    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,  # 不立即分词
        add_generation_prompt=True,  # 添加生成提示
    )
    # 设置解码超参数
    sampling_params = SamplingParams(
        temperature=0.6,  # 可选：设置温度
        seed=42,
        top_p=0.95,       # 可选：设置核采样概率
        repetition_penalty=1.05,  # 可选：设置重复惩罚
        max_tokens=30000   # 设置最大生成token数
    )

    # 生成输出
    outputs = llm.generate([text], sampling_params)

    response = [output.outputs[0].text for output in outputs]

    return response



def find_json_files(path):
    """递归查找目录中的所有JSON文件并按文件名排序"""
    if os.path.isfile(path):
        return [path] if path.endswith('.json') else []
    
    json_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    # 按文件名排序（不包含路径）
    return sorted(json_files, key=lambda x: os.path.basename(x))

def extract_base_name(filename):
    """从文件名中提取基础名称，处理带annotation后缀的情况"""
    # 移除常见的标注后缀（如_annotation, _result, _anno等）
    base_name = os.path.splitext(os.path.basename(filename))[0]
    base_name = re.sub(r'(_annotation|_result|_anno|_labeled)$', '', base_name)
    return base_name

def build_file_map(files):
    """创建文件名到文件路径的映射"""
    file_map = {}
    for file in files:
        base_name = extract_base_name(file)
        if base_name in file_map:
            print(f"警告: 基础名称 '{base_name}' 对应多个文件")
            print(f"  - {file_map[base_name]}")
            print(f"  - {file}")
        file_map[base_name] = file
    return file_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate annotation results against groundtruth.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--annotation", type=str, required=True, help="Path to the annotation result file or directory")
    parser.add_argument("--groundtruth", type=str, required=True, help="Path to the groundtruth file or directory")
    parser.add_argument("--output_path", required=True, help="Path to save score results")
    args = parser.parse_args()

    # 处理标注结果路径
    anno_path = args.annotation
    if not os.path.exists(anno_path):
        print(f"错误：标注路径 '{anno_path}' 不存在")
        sys.exit(1)
    
    # 查找所有JSON文件并分组
    if os.path.isfile(anno_path):
        groups = {"": [anno_path]}  # 单个文件作为根组
    else:
        groups = {}
        # 查找直接子目录并按名称排序
        for entry in sorted(os.scandir(anno_path), key=lambda e: e.name):
            if entry.is_dir():
                json_files = find_json_files(entry.path)
                if json_files:
                    groups[entry.name] = json_files
        
        # 如果没有找到子目录，处理当前目录
        if not groups:
            json_files = find_json_files(anno_path)
            if json_files:
                groups[""] = json_files
    
    if not groups:
        print(f"错误：标注路径 '{anno_path}' 中没有找到JSON文件")
        sys.exit(1)

    # 处理标准答案路径
    gt_path = args.groundtruth
    if not os.path.exists(gt_path):
        print(f"错误：标准答案路径 '{gt_path}' 不存在")
        sys.exit(1)
    
    gt_files = find_json_files(gt_path)
    if not gt_files:
        print(f"错误：标准答案路径 '{gt_path}' 中没有找到JSON文件")
        sys.exit(1)
    
    # 构建标准答案文件名映射
    gt_map = build_file_map(gt_files)
    print(f"找到 {len(gt_map)} 个标准答案文件")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        #gpu_memory_utilization=0.95,
        tensor_parallel_size=2
    )

    total_matched = 0
    
    # 按组名排序处理
    for group_name in sorted(groups.keys()):
        anno_files = groups[group_name]
        print(f"\n处理组: {group_name if group_name else '根目录'}")
        print(f"找到 {len(anno_files)} 个标注文件")
        
        # 创建组输出目录
        group_output = os.path.join(args.output_path, group_name) if group_name else args.output_path
        os.makedirs(group_output, exist_ok=True)
        
        # 构建标注文件映射
        anno_map = build_file_map(anno_files)
        
        # 查找匹配文件
        matched_pairs = []
        for base_name, anno_file in anno_map.items():
            if base_name in gt_map:
                matched_pairs.append((anno_file, gt_map[base_name]))
        
        # 按文件名排序匹配对
        matched_pairs.sort(key=lambda x: os.path.basename(x[0]))
        
        # 打印匹配信息
        match_count = len(matched_pairs)
        total_matched += match_count
        print(f"匹配到 {match_count} 对文件:")
        for i, (anno_file, gt_file) in enumerate(matched_pairs, 1):
            print(f"[{i}] 标注: {os.path.basename(anno_file)}")
            print(f"    答案: {os.path.basename(gt_file)}")
        
        # 处理匹配的文件对
        print(f"\n开始处理 {match_count} 对匹配文件...")
        for anno_file, gt_file in matched_pairs:
            # 读取文件内容
            with open(anno_file, "r", encoding="utf-8") as f:
                anno_content = f.read()
            with open(gt_file, "r", encoding="utf-8") as f:
                gt_content = f.read()
            
            # 生成评分
            result = generate_score(anno_content, gt_content, tokenizer, llm)
            
            # 构建输出路径
            base_name = extract_base_name(gt_file)
            output_file = os.path.join(group_output, f"{base_name}.txt")
            
            # 保存结果
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result[0])
            
            print(f"已生成：{output_file}")

    print(f"\n处理完成！共匹配 {total_matched} 对文件")
    print(f"结果保存在：{args.output_path}")
