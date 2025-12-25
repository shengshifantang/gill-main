"""
布局规划器 (Layout Planner)

基于 DeepSeek-7B，使用 LoRA 微调，使其能够输出结构化布局信息。

输出格式：<obj>对象名</obj><box>[x1,y1,x2,y2]</box>
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import re
import json


def parse_layout_output(text: str) -> List[Dict]:
    """
    解析布局输出文本，提取对象和坐标
    
    输入格式：<obj>对象名</obj><box>[x1,y1,x2,y2]</box>...
    
    返回：[{"name": "对象名", "bbox": [x1, y1, x2, y2]}]
    """
    objects = []
    
    # 匹配 <obj>...</obj><box>...</box> 模式
    pattern = r'<obj>([^<]+)</obj><box>\[([^\]]+)\]</box>'
    matches = re.findall(pattern, text)
    
    for name, bbox_str in matches:
        try:
            # 解析坐标
            bbox = [float(x.strip()) for x in bbox_str.split(',')]
            if len(bbox) == 4:
                objects.append({
                    "name": name.strip(),
                    "bbox": bbox
                })
        except:
            continue
    
    return objects


def format_layout_input(prompt: str) -> str:
    """
    格式化输入 prompt 为 Instruction Tuning 格式
    
    示例：
    输入："画一只在桌子左边的猫"
    输出："用户：画一只在桌子左边的猫\n助手：<obj>猫</obj><box>[0.0,0.3,0.4,0.7]</box>"
    """
    return f"用户：{prompt}\n助手："


class LayoutPlanner(nn.Module):
    """
    布局规划器
    
    基于预训练的 LLM（如 DeepSeek-7B），通过 LoRA 微调
    学习将自然语言描述转换为结构化布局信息。
    """
    
    def __init__(self, base_model_path: str, device: str = 'cuda', use_lora: bool = True):
        """
        Args:
            base_model_path: 基础模型路径（如 DeepSeek-7B）
            device: 设备
            use_lora: 是否使用 LoRA 微调（推荐，节省显存）
        """
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # 规范化 device 参数，便于后续统一处理
        # - "cuda" 视为 "cuda:0"
        # - "cuda:0,1" 表示使用 0、1 两张卡做 tensor parallel
        if isinstance(device, str):
            if device == "cuda":
                norm_device = "cuda:0"
            else:
                norm_device = device
        else:
            norm_device = device
        self.device = norm_device
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        
        # 添加布局相关的特殊 token
        special_tokens = {
            "additional_special_tokens": ["<obj>", "</obj>", "<box>", "</box>"]
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        print(f"✓ 添加了 {num_added} 个布局特殊 token")
        
        # 加载基础模型
        print(f"📦 加载基础模型: {base_model_path}")
        # 多卡设置：
        # - 如果 device 形如 "cuda:0,1"，使用 Hugging Face 的 tensor parallel，
        #   限制权重只切到 0、1 两张卡上，并禁止 offload 到 CPU（避免吃满内存）
        # - 否则：
        #   - device == "auto" 时由 HF 自己决定（可能用到多卡+CPU）
        #   - 其他情况认为是单设备，如 "cuda:0"
        if isinstance(norm_device, str) and norm_device.startswith("cuda") and "," in norm_device:
            # 解析 GPU id 列表，例如 "cuda:0,1"
            gpu_ids = []
            for part in norm_device.split(","):
                part = part.strip()
                if ":" in part:
                    idx = int(part.split(":")[1])
                else:
                    idx = int(part)
                gpu_ids.append(idx)

            # accelerate 要求 max_memory 的 key 为整数 GPU id 或 'cpu' / 'disk'
            max_memory = {i: "22GiB" for i in gpu_ids}
            # 禁止 offload 到 CPU，尽量只用显存（如果想允许少量 offload，可以改成比如 '8GiB'）
            max_memory["cpu"] = "0GiB"

            print(f"✓ 使用多卡 tensor parallel: GPUs={gpu_ids}, max_memory={max_memory}")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
            )
            # tokenizer / 输入统一走主卡
            self.device = f"cuda:{gpu_ids[0]}"
        else:
            if norm_device == "auto":
                device_map_arg = "auto"
            else:
                # 单 GPU：显式绑定到指定卡，避免自动 offload 到 CPU
                device_map_arg = norm_device
            print(f"✓ 使用 device_map={device_map_arg}")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map_arg,
                trust_remote_code=True
            )
        
        # 调整 embedding 大小
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # 全参数微调时启用 gradient checkpointing 以节省显存
        if not use_lora and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("✓ 启用 gradient checkpointing（节省显存）")
        
        # 使用 LoRA 微调（推荐）
        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
                
                peft_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 根据模型结构调整
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                self.model = get_peft_model(self.model, peft_config)
                print("✓ 使用 LoRA 微调（参数量大幅减少）")
            except ImportError:
                print("⚠️ peft 未安装，使用全量微调（需要更多显存）")
                use_lora = False
        
        self.use_lora = use_lora
        
        # #region agent log
        import json as _json, time as _time
        if torch.cuda.is_available():
            try:
                log_device = torch.device(device) if device != "auto" else torch.device("cuda:0")
            except Exception:
                log_device = torch.device("cuda:0")
            mem_allocated = torch.cuda.memory_allocated(log_device) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(log_device) / 1024**3
            with open("/home/lxh/Project/gill-main/.cursor/debug.log", "a") as _f:
                _f.write(
                    _json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "oom_debug",
                            "hypothesisId": "H1",
                            "location": "layout_planner.py:__init__",
                            "message": "model_loaded_memory",
                            "data": {
                                "use_lora": use_lora,
                                "mem_allocated_gb": round(mem_allocated, 2),
                                "mem_reserved_gb": round(mem_reserved, 2),
                            },
                            "timestamp": int(_time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        # #endregion
        
        self.model.eval()
    
    def forward(self, input_ids, labels=None, **kwargs):
        """Forward pass for training"""
        return self.model(input_ids=input_ids, labels=labels, **kwargs)
    
    def generate_layout(self, prompt: str, max_length: int = 128,
                       temperature: float = 0.2, top_p: float = 1.0,
                       apply_refinement: bool = True) -> Dict:
        """
        生成布局规划
        
        Args:
            prompt: 输入文本（如"画一只在桌子左边的猫"）
            max_length: 最大生成长度
            temperature: 采样温度
            top_p: nucleus sampling 参数
        
        Returns:
            {
                "layout_text": "<obj>...</obj><box>...</box>",
                "objects": [{"name": "...", "bbox": [...]}]
            }
        """
        self.model.eval()
        
        # 格式化输入
        formatted_input = format_layout_input(prompt)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # 生成：推理阶段使用低温度 + greedy，减少随机性和重复
        # 使用 max_new_tokens 而不是 max_length，避免输入长度超过 max_length 的问题
        input_length = inputs['input_ids'].shape[1]
        max_new_tokens = max(max_length - input_length, 1)  # 确保至少生成 1 个 token
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码
        generated_text = self.tokenizer.decode(
            generated_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=False
        )
        
        # 提取布局部分（从 "助手：" 之后）
        if "助手：" in generated_text:
            layout_text = generated_text.split("助手：")[-1].strip()
        else:
            layout_text = generated_text.strip()

        # 清理可能残留的 BOS token（如 DeepSeek 的 <｜begin▁of▁sentence｜>）
        bos = getattr(self.tokenizer, "bos_token", None)
        if isinstance(bos, str) and bos in layout_text:
            layout_text = layout_text.replace(bos, "").strip()
        
        # 解析对象和坐标
        objects = parse_layout_output(layout_text)

        # 根据中文位置词对 bbox 做一次启发式“吸附修正”，增强左/右/上/下等方向一致性
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
        model_path = "./model/deepseek-llm-7b-base"
    
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


def train_layout_planner(planner: LayoutPlanner, train_loader, 
                        optimizer, num_epochs: int = 3, device: str = 'cuda'):
    """
    训练布局规划器（Instruction Tuning）
    
    训练数据格式：
    {
        "input": "画一只在桌子左边的猫",
        "output": "<obj>猫</obj><box>[0.0,0.3,0.4,0.7]</box>"
    }
    """
    planner.model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # 格式化输入输出
            inputs = [format_layout_input(item["input"]) for item in batch]
            targets = [item["output"] for item in batch]
            
            # Tokenize
            # 对于多 GPU（device_map="auto"），需要确定输入应该放在哪个设备
            # 通常放在第一个 GPU 或模型的第一个设备
            if device == "auto":
                # 多 GPU 模式：找到第一个设备
                if hasattr(planner.model, 'hf_device_map') and planner.model.hf_device_map:
                    # hf_device_map 的格式可能是 {"layer_name": device_index} 或 {"layer_name": "cuda:0"}
                    first_device_value = list(planner.model.hf_device_map.values())[0]
                    if isinstance(first_device_value, torch.device):
                        input_device = first_device_value
                    elif isinstance(first_device_value, str):
                        input_device = torch.device(first_device_value)
                    elif isinstance(first_device_value, int):
                        # 设备索引，如 0, 1
                        input_device = torch.device(f"cuda:{first_device_value}")
                    else:
                        input_device = torch.device("cuda:0")
                else:
                    # 回退到 cuda:0
                    input_device = torch.device("cuda:0")
            elif isinstance(device, str) and device.startswith("cuda"):
                # 单 GPU 模式，如 "cuda:0"
                input_device = torch.device(device)
            else:
                # 其他情况（如 torch.device 对象）
                input_device = device if isinstance(device, torch.device) else torch.device(device)
            
            input_encodings = planner.tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(input_device)
            
            target_encodings = planner.tokenizer(
                targets,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(input_device)
            
            # 拼接输入和输出（用于 causal LM 训练）
            input_ids = torch.cat([input_encodings.input_ids, target_encodings.input_ids], dim=1)
            labels = input_ids.clone()
            # 只对输出部分计算 loss
            labels[:, :input_encodings.input_ids.shape[1]] = -100
            
            # #region agent log
            import json as _json, time as _time
            if torch.cuda.is_available():
                try:
                    # 确保 log_device 是一个有效的 cuda device
                    if isinstance(input_device, torch.device) and input_device.type == "cuda":
                        log_device = input_device
                    else:
                        log_device = torch.device("cuda:0")
                    mem_before_forward = torch.cuda.memory_allocated(log_device) / 1024**3
                    seq_len = input_ids.shape[1]
                    with open("/home/lxh/Project/gill-main/.cursor/debug.log", "a") as _f:
                        _f.write(
                            _json.dumps(
                                {
                                    "sessionId": "debug-session",
                                    "runId": "oom_debug",
                                    "hypothesisId": "H2",
                                    "location": "layout_planner.py:train_layout_planner",
                                    "message": "before_forward",
                                    "data": {
                                        "batch_size": len(batch),
                                        "seq_len": int(seq_len),
                                        "log_device": str(log_device),
                                        "mem_allocated_gb": round(mem_before_forward, 2),
                                    },
                                    "timestamp": int(_time.time() * 1000),
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
            # #endregion
            
            # Forward
            outputs = planner.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # #region agent log
            if torch.cuda.is_available():
                try:
                    # 确保 log_device 是一个有效的 cuda device
                    if isinstance(input_device, torch.device) and input_device.type == "cuda":
                        log_device = input_device
                    else:
                        log_device = torch.device("cuda:0")
                    mem_after_forward = torch.cuda.memory_allocated(log_device) / 1024**3
                    with open("/home/lxh/Project/gill-main/.cursor/debug.log", "a") as _f:
                        _f.write(
                            _json.dumps(
                                {
                                    "sessionId": "debug-session",
                                    "runId": "oom_debug",
                                    "hypothesisId": "H3",
                                    "location": "layout_planner.py:train_layout_planner",
                                    "message": "after_forward_before_backward",
                                    "data": {
                                        "log_device": str(log_device),
                                        "mem_allocated_gb": round(mem_after_forward, 2),
                                        "loss": float(loss.item()),
                                    },
                                    "timestamp": int(_time.time() * 1000),
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
            # #endregion
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    planner.model.eval()
    return planner
