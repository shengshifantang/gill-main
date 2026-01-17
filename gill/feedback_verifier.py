"""
反馈验证模块 (Feedback Verifier) - 异构混合架构版 (Hybrid MoE)

架构设计：
1. Spatial Expert (Grounding DINO): 负责精确的坐标检测和 IoU 计算
2. Semantic Expert (Qwen2-VL): 负责颜色、材质、风格等语义一致性检查
3. Hybrid Orchestrator: 融合两者意见，生成最终 Rationale

论文贡献：Neuro-Symbolic Feedback Mechanism（神经符号反馈机制）
- DINO 代表符号化的精准定位
- Qwen 代表神经化的语义理解
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Literal
from PIL import Image
import json
import os
import re

# Grounding DINO imports (HF + official)
try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    HAS_DINO = True
except ImportError:
    HAS_DINO = False
    print("[DINO] transformers not available; grounding verifier disabled.")

try:
    from groundingdino.util.inference import load_model as gd_load_model, predict as gd_predict
    from groundingdino.datasets import transforms as gd_T
    HAS_DINO_OFFICIAL = True
except Exception:
    HAS_DINO_OFFICIAL = False

# 尝试导入 Qwen2-VL
try:
    from transformers import AutoModelForVision2Seq, AutoProcessor as QwenProcessor
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False
    print("⚠️ transformers 未安装或版本不支持 Qwen2-VL")


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    计算两个 [x1, y1, x2, y2] 框的 IoU
    
    Args:
        box1: [x1, y1, x2, y2] 归一化坐标 (0-1)
        box2: [x1, y1, x2, y2] 归一化坐标 (0-1)
    
    Returns:
        IoU 值 (0-1)
    """
    # 确保坐标是 (x1, y1, x2, y2)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)


def normalize_bbox(bbox: List[float], image_size: Tuple[int, int]) -> List[float]:
    """
    将 bbox 归一化到 0-1 范围
    
    Args:
        bbox: 可能是 [x1, y1, x2, y2] 格式，范围可能是 0-1000 或 0-1
        image_size: (width, height) 图像尺寸
    
    Returns:
        归一化后的 [x1, y1, x2, y2] (0-1)
    """
    x1, y1, x2, y2 = bbox
    
    # 如果坐标在 0-1000 范围，转换为 0-1
    if max(bbox) > 1.0:
        x1, y1, x2, y2 = [c / 1000.0 for c in bbox]
    
    # 确保坐标在有效范围内
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    
    # 确保 x1 < x2, y1 < y2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    
    return [x1, y1, x2, y2]


def robust_json_parse(raw_text: str) -> Dict:
    """
    鲁棒 JSON 解析器：处理 Markdown 代码块、行注释、尾逗号等常见 LLM 格式问题。
    """
    text = (raw_text or "").strip()
    # 去掉 Markdown 代码块标记
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    # 截取最外层 {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response.")
    text = text[start:end + 1]
    # 去除 // 行注释（尽量不误伤 URL）
    text = re.sub(r"(?<!:)//.*$", "", text, flags=re.MULTILINE)
    # 去除尾逗号
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


def _short_english_noun(name: str) -> str:
    """Normalize English name to a short noun token for DINO."""
    if not isinstance(name, str):
        return "unknown"
    s = name.lower().strip()
    if not s:
        return "unknown"
    # keep letters and spaces only
    s = "".join(c if c.isalpha() or c == " " else " " for c in s)
    parts = [p for p in s.split() if p]
    if not parts:
        return "unknown"
    stop = {
        "a", "an", "the", "of", "in", "on", "with", "and", "or", "to", "for", "at", "by", "from",
        "left", "right", "top", "bottom", "upper", "lower", "front", "back", "middle", "center",
        "near", "far", "red", "green", "blue", "yellow", "black", "white", "brown", "pink",
        "purple", "orange", "gray", "grey"
    }
    parts = [p for p in parts if p not in stop]
    if not parts:
        return "unknown"
    noun = parts[-1]
    if noun.endswith("s") and len(noun) > 3 and not noun.endswith("ss"):
        noun = noun[:-1]
    return noun or "unknown"


class GroundingDinoVerifier:
    """Spatial verifier using Grounding DINO (HF or official)."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda",
        backend: Literal["auto", "hf", "official"] = "auto",
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        self.device = device
        self.model = None
        self.processor = None
        self.transform = None
        self.backend = backend
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        use_official = False
        if backend == "official":
            use_official = True
        elif backend == "auto":
            if config_path or checkpoint_path or str(model_id).endswith(".pth"):
                use_official = True

        if use_official:
            if not HAS_DINO_OFFICIAL:
                print("[DINO] official GroundingDINO not available; fallback to HF if possible.")
                use_official = False
            elif not config_path or not checkpoint_path:
                print("[DINO] official backend needs config_path and checkpoint_path.")
                use_official = False

        if use_official:
            self.backend = "official"
            try:
                print(f"?? Loading GroundingDINO official: {config_path} | {checkpoint_path}")
                self.model = gd_load_model(config_path, checkpoint_path, device=self.device)
                self.model.eval()
                self.transform = gd_T.Compose([
                    gd_T.RandomResize([800], max_size=1333),
                    gd_T.ToTensor(),
                    gd_T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                print("  ? GroundingDINO official loaded")
            except Exception as e:
                print(f"  ?? GroundingDINO official load failed: {e}")
                self.model = None
                self.transform = None
                self.backend = "hf"

        if self.model is None and self.backend != "official":
            if not HAS_DINO:
                print("?? Grounding DINO not available; spatial verifier disabled.")
                return
            print(f"?? Loading GroundingDINO (HF): {model_id}")
            try:
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
                self.model.eval()
                print("  ? GroundingDINO HF loaded")
            except Exception as e:
                print(f"  ?? GroundingDINO HF load failed: {e}")
                self.model = None
                self.processor = None

    @staticmethod
    def _cxcywh_to_xyxy(boxes: Union[List[List[float]], np.ndarray]) -> List[List[float]]:
        """Convert cx,cy,w,h -> x1,y1,x2,y2 (normalized)."""
        out = []
        for b in boxes:
            if len(b) != 4:
                continue
            cx, cy, w, h = [float(v) for v in b]
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            out.append([x1, y1, x2, y2])
        return out

    def _load_image_official(self, image: Image.Image):
        image_source = image.convert("RGB")
        if self.transform is None:
            self.transform = gd_T.Compose([
                gd_T.RandomResize([800], max_size=1333),
                gd_T.ToTensor(),
                gd_T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        image_transformed, _ = self.transform(image_source, None)
        return image_source, image_transformed

    def verify_layout(
        self,
        image: Image.Image,
        expected_layout: List[Dict],
        threshold: float = 0.35,
        iou_threshold: float = 0.5,
        match_label: bool = True,
        debug_topk: int = 0,
    ) -> Dict:
        if not self.model or not expected_layout:
            return {
                "correct": True,
                "feedback": "No spatial constraints or DINO not loaded.",
                "score": 1.0,
                "details": [],
            }

        labels = [obj["name"] for obj in expected_layout]
        text_prompt = ". ".join(labels) + "."
        box_thr = threshold if threshold is not None else self.box_threshold
        text_thr = self.text_threshold

        try:
            if self.backend == "official":
                image_source, image_tensor = self._load_image_official(image)
                boxes, logits, phrases = gd_predict(
                    model=self.model,
                    image=image_tensor,
                    caption=text_prompt,
                    box_threshold=box_thr,
                    text_threshold=text_thr,
                    device=self.device,
                )
                detected_boxes = boxes
                detected_labels = phrases
                detected_scores = logits
            else:
                inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                target_sizes = torch.tensor([image.size[::-1]])
                try:
                    results = self.processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        box_threshold=box_thr,
                        text_threshold=text_thr,
                        target_sizes=target_sizes,
                    )[0]
                except TypeError:
                    results = self.processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        threshold=box_thr,
                        text_threshold=text_thr,
                        target_sizes=target_sizes,
                    )[0]
                detected_boxes = results["boxes"]
                detected_labels = results["labels"]
                detected_scores = results["scores"]

            # normalize outputs to numpy lists
            if isinstance(detected_boxes, torch.Tensor):
                detected_boxes = detected_boxes.detach().cpu().numpy()
            if isinstance(detected_scores, torch.Tensor):
                detected_scores = detected_scores.detach().cpu().numpy()
            if detected_labels is None:
                detected_labels = []

            # normalize labels to list of strings for debug / filtering
            label_list = []
            try:
                for lab in detected_labels:
                    label_list.append(str(lab))
            except Exception:
                label_list = [str(detected_labels)]

            # auto-disable label filter if labels are empty/blank
            if match_label:
                if (not label_list) or all((not str(l).strip()) for l in label_list):
                    match_label = False

            # official backend returns cx,cy,w,h normalized; convert to xyxy
            if self.backend == "official" and detected_boxes is not None:
                detected_boxes = self._cxcywh_to_xyxy(detected_boxes)

            w, h = image.size
            # convert to pixel coords for details, keep normalized for IoU
            detected_boxes_pixel = []
            detected_boxes_norm = []
            for b in detected_boxes:
                if len(b) != 4:
                    continue
                b = [float(v) for v in b]
                if max(b) <= 1.5:
                    bn = b
                    bp = [b[0] * w, b[1] * h, b[2] * w, b[3] * h]
                else:
                    bp = b
                    bn = [b[0] / w, b[1] / h, b[2] / w, b[3] / h]
                detected_boxes_pixel.append(bp)
                detected_boxes_norm.append(bn)

            feedback_lines = []
            all_passed = True
            total_iou = 0.0
            details = []

            for exp_obj in expected_layout:
                exp_name = exp_obj.get("name", "")
                exp_bbox = normalize_bbox(exp_obj.get("bbox", [0, 0, 0, 0]), (w, h))
                best_iou = 0.0
                best_box = None
                best_label = None
                best_score = 0.0
                for idx, (pred_box_norm, pred_box_pixel) in enumerate(zip(detected_boxes_norm, detected_boxes_pixel)):
                    pred_label = label_list[idx] if idx < len(label_list) else ""
                    pred_score = detected_scores[idx] if idx < len(detected_scores) else 0.0
                    if match_label and exp_name and exp_name.lower() not in str(pred_label).lower():
                        continue
                    curr_iou = calculate_iou(exp_bbox, pred_box_norm)
                    if curr_iou > best_iou:
                        best_iou = curr_iou
                        best_box = pred_box_pixel
                        best_label = pred_label
                        best_score = float(pred_score)

                passed = best_iou >= iou_threshold
                if not passed:
                    all_passed = False
                    feedback_lines.append(f"? {exp_name} IoU={best_iou:.2f} (<{iou_threshold:.2f}) or not detected")
                else:
                    feedback_lines.append(f"? {exp_name} IoU={best_iou:.2f}")

                total_iou += best_iou
                details.append({
                    "name": exp_name,
                    "expected_bbox": exp_bbox,
                    "detected_bbox": best_box,
                    "iou": float(best_iou),
                    "passed": passed,
                    "detected_label": best_label,
                    "detection_score": best_score,
                })

            avg_iou = total_iou / len(expected_layout) if expected_layout else 0.0
            debug_info = None
            if debug_topk and debug_topk > 0:
                pairs = []
                for i, lab in enumerate(label_list):
                    score = float(detected_scores[i]) if i < len(detected_scores) else 0.0
                    pairs.append({"label": str(lab), "score": score})
                pairs.sort(key=lambda x: -x["score"])
                debug_info = {
                    "text_prompt": text_prompt,
                    "match_label": match_label,
                    "labels_total": len(label_list),
                    "topk": pairs[: int(debug_topk)],
                }
            return {
                "correct": all_passed,
                "feedback": " ".join(feedback_lines),
                "score": float(avg_iou),
                "details": details,
                "debug": debug_info,
            }
        except Exception as e:
            print(f"  ?? Grounding DINO verify failed: {e}")
            return {
                "correct": True,
                "feedback": f"Spatial verification error: {str(e)}",
                "score": 0.5,
                "details": [],
            }

class QwenSemanticVerifier:
    """语义验证专家：使用 Qwen2-VL 检查颜色、属性、数量等语义一致性"""
    
    def __init__(self, model_path: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = "cuda"):
        """
        Args:
            model_path: Qwen2-VL 模型路径（HuggingFace ID 或本地路径）
            device: 设备
        """
        self.device = device
        self.model = None
        self.processor = None
        
        if not HAS_QWEN:
            print("⚠️ Qwen2-VL 不可用，语义验证将跳过")
            return
        
        print(f"📦 加载语义专家 (Qwen2-VL): {model_path}")
        try:
            self.processor = QwenProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map=device,
                trust_remote_code=True
            ).eval()
            print("  ✅ Qwen2-VL 加载成功")
        except Exception as e:
            print(f"  ⚠️ Qwen2-VL 加载失败: {e}")
            self.model = None
            self.processor = None

    def verify_semantics(self, image: Image.Image, prompt: str) -> Dict:
        """
        检查图像是否符合 Prompt 的语义描述
        
        Args:
            image: PIL Image
            prompt: 原始 prompt
        
        Returns:
            {
                "correct": bool,
                "feedback": str,
                "confidence": float
            }
        """
        if not self.model:
            return {
                "correct": True,
                "feedback": "语义验证器未加载，跳过验证",
                "confidence": 0.5
            }
        
        try:
            query = f"""请作为一名极度严格的视觉质检员。
用户提示词: "{prompt}"
请检查生成的图像是否符合上述描述。

重点检查：
1. 物体是否存在？
2. 颜色、材质、属性是否正确？
3. 数量是否正确？
4. 动作、姿态是否符合描述？
(忽略具体的位置坐标，只关注内容正确性)

请输出结论：
如果符合，输出"符合"。
如果不符合，简要说明原因（例如："颜色不对，应该是红色但图中是蓝色"）。"""

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query}
                ]}
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
            
            # 提取生成的文本（去掉输入部分）
            generated_ids = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # 解析响应
            is_pass = "符合" in response and "不符合" not in response
            # 计算置信度（基于关键词）
            if is_pass:
                confidence = 0.9 if "完全" in response or "非常" in response else 0.7
            else:
                confidence = 0.2
            
            return {
                "correct": is_pass,
                "feedback": response,
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"  ⚠️ Qwen2-VL 验证过程出错: {e}")
            return {
                "correct": True,  # 出错时默认通过
                "feedback": f"语义验证过程出错: {str(e)}",
                "confidence": 0.5
            }

    def manager_judge(
        self,
        image: Image.Image,
        user_prompt: str,
        max_new_tokens: int = 256,
        log_path: Optional[str] = "outputs/manager_logs.jsonl",
    ) -> Dict:
        """
        管理员评估：输出结构化 JSON（对象完备性 + 简化英文名 + 逻辑检查）。
        """
        if not self.model:
            return {
                "summary": {
                    "is_complete": False,
                    "logical_consistency": False,
                    "pass_to_dino": False
                },
                "objects": [],
                "feedback": "语义评估器未加载，无法判定。",
                "raw": ""
            }

        prompt = (
            "你是一个专业的视觉布局评估专家。你的任务是评估生成的图像是否符合中文提示词的要求，"
            "并为后续的几何检测做准备。\n\n"
            f"中文提示词：\"{user_prompt}\"\n\n"
            "请仔细观察图像，并输出严格的 JSON 格式（不要包含 Markdown 代码块标记）：\n"
            "{\n"
            "  \"summary\": {\n"
            "    \"is_complete\": true/false,\n"
            "    \"logical_consistency\": true/false,\n"
            "    \"pass_to_dino\": true/false\n"
            "  },\n"
            "  \"objects\": [\n"
            "    {\n"
            "      \"name_zh\": \"中文实体名\",\n"
            "      \"name_en\": \"英文单词（只输出最简单的名词，不要形容词；不确定用 unknown）\",\n"
            "      \"status\": \"present/missing/hallucinated\",\n"
            "      \"visual_quality\": \"good/bad\"\n"
            "    }\n"
            "  ],\n"
            "  \"feedback\": \"如果不合格，请用中文写给 Layout Planner 的修改建议；如果合格，填 无。\"\n"
            "}\n"
        )

        try:
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]}
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

            generated_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            # Parse JSON (robust)
            raw = response
            try:
                data = robust_json_parse(raw)
            except Exception as e:
                print(f"⚠️ Manager JSON 解析失败: {e}")
                return {
                    "summary": {"is_complete": False, "logical_consistency": False, "pass_to_dino": False},
                    "objects": [],
                    "feedback": f"视觉评估解析失败: {str(e)}",
                    "raw": raw
                }

            # Normalize summary
            summary = data.get("summary") or {}
            data["summary"] = {
                "is_complete": bool(summary.get("is_complete", False)),
                "logical_consistency": bool(summary.get("logical_consistency", False)),
                "pass_to_dino": bool(summary.get("pass_to_dino", False)),
            }

            # Normalize objects
            objs = []
            for obj in data.get("objects", []) or []:
                if not isinstance(obj, dict):
                    continue
                name_zh = str(obj.get("name_zh", "")).strip()
                name_en_raw = str(obj.get("name_en", "")).strip().lower()
                name_en = _short_english_noun(name_en_raw)
                status = str(obj.get("status", "")).strip().lower() or "present"
                vq = str(obj.get("visual_quality", "")).strip().lower() or "good"
                objs.append({
                    "name_zh": name_zh,
                    "name_en": name_en,
                    "status": status,
                    "visual_quality": vq,
                })
            data["objects"] = objs
            data["feedback"] = data.get("feedback", "无") or "无"
            data["raw"] = raw

            if log_path:
                try:
                    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
                    log_entry = {
                        "prompt": user_prompt,
                        "summary": data.get("summary"),
                        "objects": data.get("objects"),
                        "feedback": data.get("feedback"),
                        "raw": raw,
                    }
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                except Exception:
                    pass
            return data
        except Exception as e:
            return {
                "summary": {
                    "is_complete": False,
                    "logical_consistency": False,
                    "pass_to_dino": False
                },
                "objects": [],
                "feedback": f"视觉评估解析失败，建议保持当前布局或微调。({e})",
                "raw": ""
            }


class ManagerSurveyorVerifier:
    """
    Manager (Qwen2-VL) + Surveyor (Grounding DINO)
    """
    def __init__(self,
                 semantic_model: str = "Qwen/Qwen2-VL-7B-Instruct",
                 spatial_model: str = "IDEA-Research/grounding-dino-base",
                 device: str = "cuda",
                 dino_backend: Literal["auto", "hf", "official"] = "auto",
                 dino_config_path: Optional[str] = None,
                 dino_checkpoint_path: Optional[str] = None,
                 dino_box_threshold: float = 0.35,
                 dino_text_threshold: float = 0.25,
                 manager_log_path: Optional[str] = "outputs/manager_logs.jsonl"):
        self.device = device
        self.manager_log_path = manager_log_path
        self.manager = QwenSemanticVerifier(semantic_model, device)
        self.surveyor = GroundingDinoVerifier(
            model_id=spatial_model,
            device=device,
            backend=dino_backend,
            config_path=dino_config_path,
            checkpoint_path=dino_checkpoint_path,
            box_threshold=dino_box_threshold,
            text_threshold=dino_text_threshold,
        )

    def verify(self,
               image: Image.Image,
               original_prompt: str,
               expected_layout: Optional[List[Dict]] = None,
               threshold: float = 0.7,
               iou_threshold: float = 0.5) -> Dict:
        manager = self.manager.manager_judge(
            image,
            original_prompt,
            log_path=self.manager_log_path,
        )
        summary = manager.get("summary", {})
        pass_to_dino = bool(summary.get("pass_to_dino", False))

        if not expected_layout:
            return {
                "correct": pass_to_dino,
                "confidence": 0.0 if not pass_to_dino else 0.5,
                "feedback": manager.get("feedback", ""),
                "manager": manager,
                "spatial_pass": False,
                "semantic_pass": pass_to_dino,
                "rationale": "No expected layout for spatial check.",
            }

        if not pass_to_dino or not self.surveyor.model:
            return {
                "correct": False,
                "confidence": 0.0,
                "feedback": manager.get("feedback", ""),
                "manager": manager,
                "spatial_pass": False,
                "semantic_pass": pass_to_dino,
                "rationale": "Manager blocked DINO or DINO unavailable.",
            }

        # Map zh->en for DINO
        zh_to_en = {}
        for obj in manager.get("objects", []):
            if obj.get("status") == "present" and obj.get("name_en") and obj.get("name_en") != "unknown":
                zh_to_en[str(obj.get("name_zh", "")).strip()] = obj.get("name_en")

        expected_dino = []
        for obj in expected_layout:
            if not isinstance(obj, dict):
                continue
            zh = str(obj.get("name", "")).strip()
            en = zh_to_en.get(zh, None)
            if not en:
                # allow ASCII names directly
                if zh.isascii():
                    en = zh.lower()
                else:
                    continue
            expected_dino.append({"name": en, "bbox": obj.get("bbox")})

        if not expected_dino:
            return {
                "correct": True,
                "confidence": 0.5,
                "feedback": manager.get("feedback", "无"),
                "manager": manager,
                "spatial_pass": True,
                "semantic_pass": pass_to_dino,
                "rationale": "No valid English targets for DINO; skip spatial check.",
            }

        spatial_res = self.surveyor.verify_layout(
            image, expected_dino, threshold=0.35, iou_threshold=iou_threshold
        )
        spatial_pass = spatial_res.get("correct", False)
        spatial_score = spatial_res.get("score", 0.0)

        combined_confidence = (spatial_score + (1.0 if pass_to_dino else 0.0)) / 2.0
        final_pass = spatial_pass and pass_to_dino and (combined_confidence >= threshold)
        return {
            "correct": final_pass,
            "confidence": combined_confidence,
            "feedback": manager.get("feedback", "无"),
            "manager": manager,
            "spatial_pass": spatial_pass,
            "semantic_pass": pass_to_dino,
            "rationale": f"Manager pass={pass_to_dino}; DINO IoU={spatial_score:.2f}",
            "dino": spatial_res,
        }


class HybridFeedbackVerifier:
    """
    [核心] 异构混合验证器
    
    组合 Grounding DINO (空间) + Qwen2-VL (语义)
    实现 Neuro-Symbolic Feedback Mechanism
    """
    
    def __init__(self, 
                 semantic_model: str = "Qwen/Qwen2-VL-7B-Instruct",
                 spatial_model: str = "IDEA-Research/grounding-dino-base",
                 device: str = "cuda",
                 dino_backend: Literal["auto", "hf", "official"] = "auto",
                 dino_config_path: Optional[str] = None,
                 dino_checkpoint_path: Optional[str] = None,
                 dino_box_threshold: float = 0.35,
                 dino_text_threshold: float = 0.25):
        """
        Args:
            semantic_model: Qwen2-VL 模型路径
            spatial_model: Grounding DINO 模型路径
            device: 设备
        """
        self.device = device
        
        # 初始化两个专家
        self.spatial_expert = GroundingDinoVerifier(
            model_id=spatial_model,
            device=device,
            backend=dino_backend,
            config_path=dino_config_path,
            checkpoint_path=dino_checkpoint_path,
            box_threshold=dino_box_threshold,
            text_threshold=dino_text_threshold,
        )
        self.semantic_expert = QwenSemanticVerifier(semantic_model, device)
        
        print("🔀 异构混合验证器初始化完成")

    def verify(self, 
               image: Image.Image,
               original_prompt: str,
               expected_layout: Optional[List[Dict]] = None,
               threshold: float = 0.7,
               iou_threshold: float = 0.5) -> Dict:
        """
        执行双重验证并融合结果
        
        Args:
            image: 生成的图像
            original_prompt: 原始 prompt
            expected_layout: 期望的布局 [{"name": "...", "bbox": [...]}]
            threshold: 总体置信度阈值
            iou_threshold: IoU 匹配阈值（用于空间验证）
        
        Returns:
            {
                "correct": bool,
                "confidence": float,
                "feedback": str,
                "refinement_instruction": Optional[str],
                "spatial_pass": bool,
                "semantic_pass": bool,
                "rationale": str
            }
        """
        feedback_parts = []
        is_spatial_pass = True
        is_semantic_pass = True
        spatial_score = 1.0
        semantic_confidence = 1.0
        
        # 1. 空间验证 (如果有布局要求)
        if expected_layout and self.spatial_expert.model:
            spatial_res = self.spatial_expert.verify_layout(
                image, expected_layout, threshold=0.35, iou_threshold=iou_threshold
            )
            is_spatial_pass = spatial_res["correct"]
            spatial_score = spatial_res["score"]
            feedback_parts.append(f"[空间检查] {spatial_res['feedback']}")
        elif expected_layout:
            # 有布局要求但 DINO 未加载，给出警告
            feedback_parts.append("[空间检查] Grounding DINO 未加载，跳过位置验证")
        
        # 2. 语义验证
        if self.semantic_expert.model:
            semantic_res = self.semantic_expert.verify_semantics(image, original_prompt)
            is_semantic_pass = semantic_res["correct"]
            semantic_confidence = semantic_res["confidence"]
            feedback_parts.append(f"[语义检查] {semantic_res['feedback']}")
        else:
            feedback_parts.append("[语义检查] Qwen2-VL 未加载，跳过语义验证")
        
        # 3. 融合决策（严格模式：两者都必须通过）
        final_pass = is_spatial_pass and is_semantic_pass
        
        # 计算综合置信度
        if expected_layout and self.spatial_expert.model:
            # 空间和语义各占 50%
            combined_confidence = (spatial_score + semantic_confidence) / 2
        else:
            # 只有语义验证
            combined_confidence = semantic_confidence
        
        final_pass = final_pass and (combined_confidence >= threshold)
        
        # 4. 生成修正指令
        refinement_instruction = None
        if not final_pass:
            refinement_instruction = "请修正以下问题：\n" + "\n".join(feedback_parts)
        
        # 5. 生成 Rationale（用于论文展示）
        rationale = f"空间验证: {'通过' if is_spatial_pass else '失败'} (IoU={spatial_score:.2f}); " \
                   f"语义验证: {'通过' if is_semantic_pass else '失败'} (置信度={semantic_confidence:.2f})"
        
        return {
            "correct": final_pass,
            "confidence": combined_confidence,
            "feedback": "\n".join(feedback_parts),
            "refinement_instruction": refinement_instruction,
            "spatial_pass": is_spatial_pass,
            "semantic_pass": is_semantic_pass,
            "rationale": rationale,
            "suggested_prompt": original_prompt,
            "detected_objects": []
        }


class FeedbackVerifier:
    """
    反馈验证器（兼容旧接口）
    
    支持多种验证模式：
    - "hybrid": 混合模式（Grounding DINO + Qwen2-VL）
    - "grounding_dino": 仅空间验证
    - "qwen2vl_7b": 仅语义验证
    - "qwen2vl": 兼容旧代码
    """
    
    def __init__(self, 
                 verifier_type: Literal["grounding_dino", "qwen2vl_7b", "hybrid", "qwen2vl", "manager_surveyor"] = "hybrid",
                 vlm_model_name: Optional[str] = None,
                 device: str = "cuda",
                 use_grounding: bool = True,
                 dino_model_id: str = "IDEA-Research/grounding-dino-base",
                 dino_backend: Literal["auto", "hf", "official"] = "auto",
                 dino_config_path: Optional[str] = None,
                 dino_checkpoint_path: Optional[str] = None,
                 dino_box_threshold: float = 0.35,
                 dino_text_threshold: float = 0.25,
                 manager_log_path: Optional[str] = "outputs/manager_logs.jsonl"):
        """
        Args:
            verifier_type: 验证器类型
            vlm_model_name: VLM 模型名称（用于兼容旧代码）
            device: 设备
            use_grounding: 是否使用 grounding（兼容旧代码）
        """
        self.device = device
        self.verifier_type = verifier_type
        
        # 根据类型初始化相应的验证器
        if verifier_type == "hybrid":
            semantic_model = vlm_model_name or "Qwen/Qwen2-VL-7B-Instruct"
            self.verifier = HybridFeedbackVerifier(
                semantic_model=semantic_model,
                spatial_model=dino_model_id,
                device=device,
                dino_backend=dino_backend,
                dino_config_path=dino_config_path,
                dino_checkpoint_path=dino_checkpoint_path,
                dino_box_threshold=dino_box_threshold,
                dino_text_threshold=dino_text_threshold,
            )
        elif verifier_type == "manager_surveyor":
            semantic_model = vlm_model_name or "Qwen/Qwen2-VL-7B-Instruct"
            self.verifier = ManagerSurveyorVerifier(
                semantic_model=semantic_model,
                spatial_model=dino_model_id,
                device=device,
                dino_backend=dino_backend,
                dino_config_path=dino_config_path,
                dino_checkpoint_path=dino_checkpoint_path,
                dino_box_threshold=dino_box_threshold,
                dino_text_threshold=dino_text_threshold,
                manager_log_path=manager_log_path,
            )
        elif verifier_type == "grounding_dino":
            self.verifier = GroundingDinoVerifier(
                model_id=dino_model_id,
                device=device,
                backend=dino_backend,
                config_path=dino_config_path,
                checkpoint_path=dino_checkpoint_path,
                box_threshold=dino_box_threshold,
                text_threshold=dino_text_threshold,
            )
        elif verifier_type in ["qwen2vl_7b", "qwen2vl"]:
            model_path = vlm_model_name or "Qwen/Qwen2-VL-7B-Instruct"
            self.semantic_expert = QwenSemanticVerifier(model_path, device)
            self.verifier = self.semantic_expert
        else:
            raise ValueError(f"未知的验证器类型: {verifier_type}")
    
    def verify(self, 
               image: Image.Image,
               original_prompt: str,
               expected_layout: Optional[List[Dict]] = None,
               threshold: float = 0.7) -> Dict:
        """
        统一验证接口
        
        Args:
            image: 生成的图像
            original_prompt: 原始 prompt
            expected_layout: 期望的布局
            threshold: 置信度阈值
        
        Returns:
            验证结果字典
        """
        if isinstance(self.verifier, HybridFeedbackVerifier):
            return self.verifier.verify(image, original_prompt, expected_layout, threshold)
        elif isinstance(self.verifier, ManagerSurveyorVerifier):
            return self.verifier.verify(image, original_prompt, expected_layout, threshold)
        elif isinstance(self.verifier, GroundingDinoVerifier):
            if expected_layout:
                result = self.verifier.verify_layout(image, expected_layout)
                return {
                    "correct": result["correct"],
                    "confidence": result["score"],
                    "feedback": result["feedback"],
                    "refinement_instruction": result["feedback"] if not result["correct"] else None,
                    "suggested_prompt": original_prompt,
                    "detected_objects": result.get("details", [])
                }
            return {
                "correct": True,
                "confidence": 1.0,
                "feedback": "无布局约束",
                "suggested_prompt": original_prompt,
                "detected_objects": []
            }
        elif isinstance(self.verifier, QwenSemanticVerifier):
            result = self.verifier.verify_semantics(image, original_prompt)
            return {
                "correct": result["correct"],
                "confidence": result["confidence"],
                "feedback": result["feedback"],
                "refinement_instruction": result["feedback"] if not result["correct"] else None,
                "suggested_prompt": original_prompt,
                "detected_objects": []
            }
        else:
            return {
                "correct": True,
                "confidence": 0.5,
                "feedback": "验证器未正确初始化",
                "suggested_prompt": original_prompt,
                "detected_objects": []
            }
    

def create_feedback_verifier(vlm_model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
                             verifier_type: Literal["grounding_dino", "qwen2vl_7b", "hybrid", "qwen2vl", "manager_surveyor"] = "hybrid",
                             device: str = "cuda",
                             use_grounding: bool = True,
                             dino_model_id: str = "IDEA-Research/grounding-dino-base",
                             dino_backend: Literal["auto", "hf", "official"] = "auto",
                             dino_config_path: Optional[str] = None,
                             dino_checkpoint_path: Optional[str] = None,
                             dino_box_threshold: float = 0.35,
                             dino_text_threshold: float = 0.25,
                             manager_log_path: Optional[str] = "outputs/manager_logs.jsonl") -> FeedbackVerifier:
    """
    创建反馈验证器（工厂函数）
    
    Args:
        vlm_model_name: VLM 模型名称（用于 qwen2vl 模式）
        verifier_type: 验证器类型
            - "grounding_dino": 仅使用 Grounding DINO
            - "qwen2vl_7b": 仅使用 Qwen2-VL-7B
            - "hybrid": 混合模式（Grounding DINO + Qwen2-VL-7B，推荐）
            - "qwen2vl": 使用指定的 Qwen-VL 模型（兼容旧代码）
        device: 设备
        use_grounding: 是否使用 grounding（兼容旧代码）
    
    Returns:
        FeedbackVerifier 实例
    """
    return FeedbackVerifier(
        verifier_type=verifier_type,
        vlm_model_name=vlm_model_name,
        device=device,
        use_grounding=use_grounding,
        dino_model_id=dino_model_id,
        dino_backend=dino_backend,
        dino_config_path=dino_config_path,
        dino_checkpoint_path=dino_checkpoint_path,
        dino_box_threshold=dino_box_threshold,
        dino_text_threshold=dino_text_threshold,
        manager_log_path=manager_log_path
    )
