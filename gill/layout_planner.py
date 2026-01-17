"""
布局规划器 (Layout Planner) - Refactored

基于 DeepSeek/Qwen，使用 LoRA 微调，通过 Instruction Tuning 输出结构化布局。
与训练脚本对齐，统一使用 Chat Template。
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import re
import json

FORMAT_INSTRUCTION = (
    "请严格按照以下格式输出：<obj>名称</obj><box>[x1,y1,x2,y2]</box>... "
    "坐标范围为0-1000，且必须覆盖提示中的所有实体，不要遗漏。"
    "如果无法给出布局，请输出 <no_layout>。只输出格式，不要解释。"
)


def _append_format_instruction(text: str) -> str:
    if "<obj>" in text or "只输出格式" in text:
        return text
    return f"{text}\n\n{FORMAT_INSTRUCTION}"


def parse_layout_output(text: str) -> List[Dict]:
    """
    解析布局输出文本，提取对象和坐标
    输入格式：<obj>对象名</obj><box>[x1,y1,x2,y2]</box>
    
    支持两种解析模式：
    1. 标准格式：<obj>...</obj><box>[...]</box>
    2. 后备格式：从乱码中提取坐标 [x1,y1,x2,y2] 和对象名
    """
    objects = []

    if "<no_layout>" in text:
        return objects

    def _normalize(text_in: str) -> str:
        # Normalize common full-width punctuation to improve parsing robustness.
        text_in = text_in.replace("，", ",").replace("；", ";")
        text_in = text_in.replace("［", "[").replace("］", "]")
        return text_in

    def _parse_bbox(bbox_str: str) -> Optional[List[float]]:
        # Extract floats to handle spaces, Chinese comma, or mixed separators.
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", bbox_str)
        if len(nums) < 4:
            return None
        bbox = [float(x) for x in nums[:4]]
        # 简单的坐标范围检查/归一化（如果是 0-1000 格式）
        if max(bbox) > 1.5:
            bbox = [b / 1000.0 for b in bbox]
        return bbox

    text = _normalize(text)

    # 模式1：标准格式解析（允许空格/换行）
    pattern = re.compile(r"<obj>\s*([^<]+?)\s*</obj>\s*<box>\s*\[([^\]]+)\]\s*</box>", re.S)
    matches = pattern.findall(text)

    if matches:  # 如果标准格式解析成功，直接返回
        for name, bbox_str in matches:
            try:
                bbox = _parse_bbox(bbox_str)
                if bbox is None or len(bbox) != 4:
                    continue
                objects.append({
                    "name": name.strip(),
                    "bbox": bbox
                })
            except Exception:
                continue
        return objects
    
    # 模式2：后备解析（从乱码中提取坐标和对象名）
    # 提取所有坐标格式：[...]
    bbox_pattern = r'\[([^\]]+)\]'
    bbox_matches = re.findall(bbox_pattern, text)
    
    # 提取可能的对象名（中文词汇，长度 1-6）
    # 清理特殊 token 和乱码字符
    cleaned_text = re.sub(r'</?tool_call>', '', text)
    cleaned_text = re.sub(r'<\|[^|]+\|>', '', cleaned_text)
    cleaned_text = re.sub(r'[a-zA-Z]{3,}', '', cleaned_text)  # 移除长英文单词（如 useRal）
    
    # 提取中文词汇作为对象名
    chinese_words = re.findall(r'[\u4e00-\u9fff]+', cleaned_text)
    
    for i, bbox_str in enumerate(bbox_matches):
        try:
            bbox = _parse_bbox(bbox_str)
            if bbox is None or len(bbox) != 4:
                continue

            # 获取对象名（优先使用对应位置的中文词，否则使用"物体"）
            name = chinese_words[i] if i < len(chinese_words) else "物体"

            objects.append({
                "name": name.strip(),
                "bbox": bbox
            })
        except Exception:
            continue
    
    return objects


def format_layout_input(tokenizer, prompt: str, enable_cot: bool = False, feedback: Optional[str] = None) -> str:
    """
    使用 Tokenizer 的 Chat Template 格式化输入
    """
    if feedback:
        user_content = f"{prompt}\n\n上一轮反馈：{feedback}\n请根据反馈调整布局。"
    elif enable_cot:
        user_content = f"""{prompt}

请按以下步骤思考并规划布局：
1. 首先，分析提示词中的空间关系（如"左边"、"上方"等）
2. 然后，确定每个物体的相对位置
3. 最后，输出布局坐标"""
    else:
        user_content = prompt

    user_content = _append_format_instruction(user_content)

    messages = [{"role": "user", "content": user_content}]
    
    # 使用 apply_chat_template，并添加 generation prompt
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return formatted_text


class LayoutPlanner(nn.Module):
    def __init__(self, base_model_path: str, device: str = 'cuda', use_lora: bool = True):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # 处理 device 参数
        self.device = device if isinstance(device, str) else "cuda"
        if self.device == "cuda": 
            self.device = "cuda:0"

        print(f"📦 加载 Tokenizer: {base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        
        # 添加布局相关的特殊 token
        special_tokens = {"additional_special_tokens": ["<obj>", "</obj>", "<box>", "</box>", "<no_layout>"]}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            print(f"✓ 添加了 {num_added} 个布局特殊 token")

        # 加载模型
        print(f"📦 加载基础模型: {base_model_path}")
        # 简单处理 device_map
        device_map = "auto" if self.device == "auto" else self.device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True
        )
        
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            
        self.use_lora = use_lora
        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
                peft_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                self.model = get_peft_model(self.model, peft_config)
                print("✓ 使用 LoRA 微调")
            except ImportError:
                print("⚠️ peft 未安装，使用全量微调")
                self.use_lora = False
        
        self.model.eval()

    def forward(self, input_ids, labels=None, **kwargs):
        """Forward pass for training"""
        return self.model(input_ids=input_ids, labels=labels, **kwargs)

    def generate_layout(self, prompt: str, max_length: int = 512,
                       temperature: float = 0.2, top_p: float = 0.9,
                       apply_refinement: bool = True, enable_cot: bool = False,
                       feedback: Optional[str] = None) -> Dict:
        """
        生成布局规划
        
        Args:
            prompt: 输入文本（如"画一只在桌子左边的猫"）
            max_length: 最大生成长度
            temperature: 采样温度
            top_p: nucleus sampling 参数
            apply_refinement: 是否应用启发式修正
            enable_cot: 是否启用 Chain-of-Thought（思考过程）
            feedback: 上一轮的反馈（用于修正）
        
        Returns:
            {
                "layout_text": "<obj>...</obj><box>...</box>",
                "objects": [{"name": "...", "bbox": [...]}]
            }
        """
        self.model.eval()
        
        # 1. 格式化输入 (使用 Chat Template)
        formatted_input = format_layout_input(
            self.tokenizer, prompt, enable_cot=enable_cot, feedback=feedback
        )
        
        # 2. Tokenize
        inputs = self.tokenizer(
            formatted_input, return_tensors="pt"
        ).to(self.model.device)
        
        # 3. 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,  # 建议开启采样以避免死循环
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 4. 解码 (只保留新生成的部分)
        input_len = inputs.input_ids.shape[1]
        generated_part = generated_ids[0][input_len:]
        layout_text = self.tokenizer.decode(generated_part, skip_special_tokens=False)
        
        # 5. 清理 (Qwen/DeepSeek 特殊 token)
        layout_text = layout_text.strip()
        for token in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
            layout_text = layout_text.replace(token, "").strip()
            
        # 6. 解析
        objects = parse_layout_output(layout_text)
        
        if apply_refinement:
            objects = refine_layout_with_caption(prompt, objects)
            
        return {
            "layout_text": layout_text,
            "objects": objects
        }
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[Dict]:
        """批量生成布局"""
        results = []
        for prompt in prompts:
            result = self.generate_layout(prompt, **kwargs)
            results.append(result)
        return results


def create_layout_planner_from_gill(gill_model, tokenizer, use_lora: bool = True) -> LayoutPlanner:
    """
    从 GILL 模型创建布局规划器
    
    使用 GILL 中的 LLM 作为基础模型
    """
    base_lm = gill_model.model.lm
    
    # 获取模型路径（如果是本地路径）
    if hasattr(base_lm, 'config') and hasattr(base_lm.config, '_name_or_path'):
        model_path = base_lm.config._name_or_path
    else:
        # 默认路径
        model_path = "./model/Qwen2.5-7B-Instruct"
    
    planner = LayoutPlanner(model_path, device=base_lm.device, use_lora=use_lora)
    
    return planner


def refine_layout_with_caption(caption: str, objects: List[Dict]) -> List[Dict]:
    """
    后处理：更强地依赖 caption 的中文位置词和名词，
    在检测到明显的「左/右/上/下/中间/左下角/右下角」结构时，
    直接用启发式规则重建一个小的 objects 列表，忽略 LM 原始 objects。
    """
    if not objects:
        objects = []

    text = str(caption)

    # 如果 caption 里没有任何位置词，就直接返回原始 objects（只做名字清洗）
    position_keywords = ["左边", "左侧", "左方", "右边", "右侧", "右方",
                         "上方", "上边", "上面", "下方", "下边", "下面",
                         "中间", "中央", "中心", "左下角", "右下角"]
    has_position = any(k in text for k in position_keywords)

    # 简单名词抽取：优先用 jieba.posseg，不行就用一个很粗糙的备选方案
    nouns: List[str] = []
    try:
        import jieba.posseg as pseg  # type: ignore

        words = pseg.cut(text)
        for w, flag in words:
            if flag.startswith("n"):  # 名词
                w = w.strip()
                if w and w not in nouns:
                    nouns.append(w)
    except Exception:
        # 简单兜底：按常见分隔符切分，取长度 1~4 的短片段
        rough_parts = []
        for seg in re.split(r"[，。、“”！!？?\s]", text):
            seg = seg.strip()
            if 1 <= len(seg) <= 4:
                rough_parts.append(seg)
        nouns = list(dict.fromkeys(rough_parts))  # 去重且保持顺序

    # 只保留前两个名词，分别当作「主/副」对象
    if nouns:
        main_name = nouns[0]
        second_name = nouns[1] if len(nouns) > 1 else None
    else:
        main_name = objects[0].get("name", "物体") if objects else "物体"
        second_name = objects[1].get("name", None) if len(objects) > 1 else None

    # 预定义槽位
    slots = {
        "left": [0.0, 0.1, 0.4, 0.9],
        "right": [0.6, 0.1, 1.0, 0.9],
        "top": [0.1, 0.0, 0.9, 0.4],
        "bottom": [0.1, 0.6, 0.9, 1.0],
        "center": [0.3, 0.3, 0.7, 0.7],
        "bottom_left": [0.0, 0.6, 0.4, 1.0],
        "bottom_right": [0.6, 0.6, 1.0, 1.0],
    }

    # 如果没有明显的位置词，就只做名字清洗，保持原结果
    if not has_position:
        for obj in objects:
            name = str(obj.get("name", "")).strip()
            if name.startswith(("是", "在", "有")) and len(name) > 1:
                name = name[1:]
            obj["name"] = name
        return objects

    # 有明显位置词时：直接根据 caption 重建 objects 列表（最多两个对象）
    new_objects: List[Dict] = []

    def add_obj(name: str, slot_key: str):
        name = name.strip() or "物体"
        # 去掉前导虚词
        if name.startswith(("是", "在", "有")) and len(name) > 1:
            name = name[1:]
        bbox = slots[slot_key]
        new_objects.append({"name": name, "bbox": bbox})

    # 默认槽位
    main_slot = "center"
    second_slot = "center"

    # 左 / 右
    if any(k in text for k in ["左边", "左侧", "左方"]):
        main_slot = "left"
    if any(k in text for k in ["右边", "右侧", "右方"]):
        second_slot = "right"

    # 上 / 下
    if any(k in text for k in ["上方", "上边", "上面"]):
        main_slot = "top"
    if any(k in text for k in ["下方", "下边", "下面"]):
        second_slot = "bottom"

    # 中间 / 中央
    if any(k in text for k in ["中间", "中央", "中心"]):
        main_slot = "center"

    # 左下角 / 右下角 优先级更高，覆盖前面的 bottom / left/right
    if "左下角" in text:
        second_slot = "bottom_left"
    if "右下角" in text:
        second_slot = "bottom_right"

    # 主对象
    add_obj(main_name, main_slot)
    # 副对象（如果有）
    if second_name is not None:
        add_obj(second_name, second_slot)

    return new_objects
