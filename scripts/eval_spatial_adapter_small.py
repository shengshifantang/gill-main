#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小规模定量评估：对比 bbox+phrase vs bbox-only 的位置/关系准确率。

输入：run_full_chain_samples.py 产生的 results.jsonl
输出：位置准确率（IoU）、关系准确率（left/right/above/below）
依赖：Grounding DINO（transformers）
"""

import argparse
import json
import os
import random
import re
import sys
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

# 允许导入 gill
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gill.feedback_verifier import GroundingDinoVerifier, normalize_bbox, QwenSemanticVerifier
from gill.layout_planner import parse_layout_output


def _dist_info() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=backend)
    return distributed, rank, world_size, local_rank


_MEASURE_RE = re.compile(r"^(一|二|三|四|五|六|七|八|九|十|两|几|多|每)?(个|只|条|张|把|台|部|辆|块|片|件|根|位|名|对|双|群)")
_NOISE_RE = re.compile(r"(正在|位于|看着|站在|坐在|躺在|趴在|穿着|拿着|走在|骑着)")


def _smart_expand_objects(objects: List[Dict]) -> List[Dict]:
    if not objects:
        return objects

    def _expand_bbox(bbox: List[float], name: str) -> List[float]:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        target_w, target_h = 0.05, 0.05
        if any(k in name for k in ["人", "男", "女", "童", "客"]):
            target_w, target_h = 0.08, 0.15
        elif any(k in name for k in ["灯", "球", "盘", "杯", "瓶", "鸟"]):
            target_w, target_h = 0.06, 0.06

        if w < target_w:
            half = target_w / 2.0
            x1 = max(0.0, cx - half)
            x2 = min(1.0, cx + half)
        if h < target_h:
            half = target_h / 2.0
            y1 = max(0.0, cy - half)
            y2 = min(1.0, cy + half)
        return [x1, y1, x2, y2]

    for obj in objects:
        bbox = obj.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        name = str(obj.get("name", ""))
        obj["bbox"] = _expand_bbox([float(v) for v in bbox], name)
    return objects


def clean_object_name(name: str, max_len: int = 10, min_len: int = 1) -> str:
    if not isinstance(name, str):
        return ""
    name = name.strip()
    if not name:
        return ""
    name = re.sub(r"<[^>]+>", "", name)
    name = re.sub(r"[\"'“”‘’（）()《》【】\[\]{}<>]", "", name)
    name = re.sub(r"[，,。\.、;；:：!?！？~`·•]", "", name)
    name = re.sub(r"\s+", "", name)
    if not name:
        return ""
    name = _MEASURE_RE.sub("", name)
    if not name:
        return ""
    if "的" in name:
        parts = [p for p in name.split("的") if p]
        if parts:
            name = parts[-1]
    if len(name) < min_len or len(name) > max_len:
        return ""
    if _NOISE_RE.search(name):
        return ""
    return name


def _center(b: List[float]) -> Tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _relation_from_gt(a: Dict, b: Dict, min_sep: float = 0.1) -> Optional[str]:
    ax, ay = _center(a["bbox"])
    bx, by = _center(b["bbox"])
    dx = ax - bx
    dy = ay - by
    if abs(dx) >= abs(dy) and abs(dx) >= min_sep:
        return "left" if dx < 0 else "right"
    if abs(dy) >= min_sep:
        return "above" if dy < 0 else "below"
    return None


def _relation_ok(rel: str, a: List[float], b: List[float], margin: float = 0.02) -> bool:
    ax, ay = _center(a)
    bx, by = _center(b)
    if rel == "left":
        return ax + margin < bx
    if rel == "right":
        return ax > bx + margin
    if rel == "above":
        return ay + margin < by
    if rel == "below":
        return ay > by + margin
    return False


def _attach_name_en(
    parsed: List[Dict],
    layout_objects: List[Dict],
    name_max_len: int,
    name_min_len: int,
) -> List[Dict]:
    if not parsed or not layout_objects:
        return parsed
    # build map from cleaned zh name -> name_en
    name_map = {}
    for obj in layout_objects:
        if not isinstance(obj, dict):
            continue
        zh = clean_object_name(obj.get("name", ""), name_max_len, name_min_len)
        en = str(obj.get("name_en", "")).strip().lower()
        if not zh or not en:
            continue
        name_map.setdefault(zh, en)

    for obj in parsed:
        if not isinstance(obj, dict):
            continue
        if obj.get("name_en"):
            continue
        zh = clean_object_name(obj.get("name", ""), name_max_len, name_min_len)
        if not zh:
            continue
        en = name_map.get(zh)
        if en:
            obj["name_en"] = en
    return parsed


def _get_expected_objects(record: Dict, name_max_len: int, name_min_len: int, smart_expand: bool) -> List[Dict]:
    layout = record.get("layout") or {}
    objects: List[Dict] = []
    layout_text = layout.get("layout_text")
    if isinstance(layout_text, str) and "<box>" in layout_text:
        try:
            objects = parse_layout_output(layout_text)
        except Exception:
            objects = []
    layout_objects = layout.get("objects") or record.get("objects") or []
    if objects:
        # attach name_en from layout_objects if available
        objects = _attach_name_en(objects, layout_objects, name_max_len, name_min_len)
    else:
        objects = layout_objects
    normed = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = clean_object_name(obj.get("name", ""), name_max_len, name_min_len)
        if not name:
            continue
        bbox = obj.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        # normalize to 0-1 if needed (parse_layout_output already normalized if used)
        b = [float(v) for v in bbox]
        if max(b) > 1.5:
            # fallback normalization for legacy stored objects
            if max(b) <= 100:
                b = [v / 100.0 for v in b]
            else:
                b = [v / 1000.0 for v in b]
        b = [max(0.0, min(1.0, v)) for v in b]
        name_en = str(obj.get("name_en", "")).strip().lower() if isinstance(obj.get("name_en", None), str) else ""
        normed.append({"name": name, "bbox": b, "name_en": name_en})
    if smart_expand:
        normed = _smart_expand_objects(normed)
    return normed


def _map_expected_with_manager(
    expected: List[Dict],
    manager: Dict,
    name_max_len: int,
    name_min_len: int,
    loose_match: bool = False,
) -> List[Dict]:
    """Map Chinese object names to English using Qwen manager output."""
    zh_to_en = {}
    manager_items = []
    for obj in manager.get("objects", []) or []:
        if not isinstance(obj, dict):
            continue
        if obj.get("status") != "present":
            continue
        name_zh_raw = str(obj.get("name_zh", "")).strip()
        name_zh = clean_object_name(name_zh_raw, name_max_len, name_min_len)
        name_en = str(obj.get("name_en", "")).strip().lower()
        if not name_zh or not name_en or name_en == "unknown":
            continue
        # simplify to single noun token
        name_en = "".join([c if c.isalpha() or c == " " else " " for c in name_en]).strip()
        if not name_en:
            continue
        if " " in name_en:
            name_en = name_en.split()[0]
        zh_to_en[name_zh] = name_en
        manager_items.append((name_zh, name_en))

    mapped = []
    for obj in expected:
        name_zh = clean_object_name(str(obj.get("name", "")).strip(), name_max_len, name_min_len)
        if not name_zh:
            continue
        name_en = zh_to_en.get(name_zh)
        if not name_en and loose_match:
            for m_zh, m_en in manager_items:
                if m_zh and (m_zh in name_zh or name_zh in m_zh):
                    name_en = m_en
                    break
        if not name_en:
            if name_zh.isascii():
                name_en = name_zh.lower()
            else:
                continue
        mapped.append({"name": name_en, "bbox": obj.get("bbox")})
    return mapped


def _load_zh_en_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"⚠️ zh-en map not found: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"⚠️ zh-en map is not a dict: {path}")
            return {}
        out = {}
        for k, v in data.items():
            kz = clean_object_name(str(k), 10, 1)
            vz = str(v).strip().lower()
            if not kz or not vz:
                continue
            out[kz] = vz
        return out
    except Exception as e:
        print(f"⚠️ zh-en map load failed: {e}")
        return {}


def _map_expected_with_zh_map(
    expected: List[Dict],
    zh_en_map: Dict[str, str],
    name_max_len: int,
    name_min_len: int,
    loose_match: bool = False,
    keep_unknown: bool = False,
) -> List[Dict]:
    if not zh_en_map:
        return []
    items = list(zh_en_map.items())
    mapped = []
    for obj in expected:
        name_zh = clean_object_name(str(obj.get("name", "")).strip(), name_max_len, name_min_len)
        if not name_zh:
            continue
        name_en = zh_en_map.get(name_zh)
        if not name_en and loose_match:
            for k, v in items:
                if k and (k in name_zh or name_zh in k):
                    name_en = v
                    break
        if not name_en:
            if name_zh.isascii():
                name_en = name_zh.lower()
            elif keep_unknown:
                name_en = name_zh
            else:
                continue
        mapped.append({"name": name_en, "bbox": obj.get("bbox")})
    return mapped


def _map_expected_with_name_en(expected: List[Dict]) -> List[Dict]:
    """Use name_en field (if provided) to build English phrases for DINO."""
    mapped = []
    for obj in expected:
        name_en = str(obj.get("name_en", "") or "").strip().lower()
        if not name_en:
            name_zh = str(obj.get("name", "")).strip()
            if name_zh.isascii():
                name_en = name_zh.lower()
            else:
                continue
        # keep only simple ascii tokens
        name_en = "".join([c if c.isalpha() or c == " " else " " for c in name_en]).strip()
        if not name_en:
            continue
        if " " in name_en:
            name_en = name_en.split()[0]
        mapped.append({"name": name_en, "bbox": obj.get("bbox")})
    return mapped


def _reservoir_sample(path: str, k: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    reservoir = []
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if not item.get("image_path"):
                continue
            seen += 1
            if len(reservoir) < k:
                reservoir.append(item)
            else:
                j = rng.randint(1, seen)
                if j <= k:
                    reservoir[j - 1] = item
    return reservoir


def evaluate_results(
    results_path: str,
    verifier: GroundingDinoVerifier,
    manager: Optional[QwenSemanticVerifier],
    use_manager: bool,
    manager_no_gate: bool,
    manager_loose_match: bool,
    zh_en_map: Dict[str, str],
    zh_en_loose_match: bool,
    zh_en_keep_unknown: bool,
    use_name_en: bool,
    dino_match_label: bool,
    debug_dino: int,
    debug_dino_topk: int,
    debug_dino_jsonl: Optional[str],
    manager_max_new_tokens: int,
    manager_log_path: Optional[str],
    max_samples: int,
    seed: int,
    dino_thresh: float,
    iou_thresh: float,
    rel_min_sep: float,
    rel_margin: float,
    name_max_len: int,
    name_min_len: int,
    distributed: bool,
    rank: int,
    world_size: int,
    smart_expand: bool,
    save_jsonl: Optional[str] = None,
):
    records = _reservoir_sample(results_path, max_samples, seed) if max_samples > 0 else []
    if not records:
        # fallback: load all
        with open(results_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

    if distributed and world_size > 1:
        records = [r for i, r in enumerate(records) if (i % world_size) == rank]

    total_samples = 0
    total_objects = 0
    detected_objects = 0
    pos_ok = 0
    iou_sum = 0.0

    rel_total = 0
    rel_matched = 0
    rel_ok = 0

    manager_total = 0
    manager_pass = 0
    dino_samples = 0

    per_sample = []
    debug_count = 0
    debug_f = None
    if debug_dino_jsonl:
        os.makedirs(os.path.dirname(debug_dino_jsonl) or ".", exist_ok=True)
        debug_f = open(debug_dino_jsonl, "w", encoding="utf-8")

    for item in records:
        img_path = item.get("image_path")
        if not img_path or not os.path.exists(img_path):
            continue
        expected = _get_expected_objects(item, name_max_len, name_min_len, smart_expand)
        if not expected:
            continue
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        manager_data = None
        if use_manager and manager is not None:
            manager_total += 1
            manager_data = manager.manager_judge(
                image,
                item.get("prompt", ""),
                max_new_tokens=manager_max_new_tokens,
                log_path=manager_log_path,
            )
            pass_to_dino = bool((manager_data.get("summary") or {}).get("pass_to_dino", False))
            if pass_to_dino:
                manager_pass += 1
            if (not pass_to_dino) and (not manager_no_gate):
                per_sample.append({
                    "image_path": img_path,
                    "objects": len(expected),
                    "pos_ok": 0,
                    "mean_iou": 0.0,
                    "rel_total": 0,
                    "rel_matched": 0,
                    "rel_ok": 0,
                    "manager_pass": False,
                })
                continue
            expected = _map_expected_with_manager(
                expected,
                manager_data,
                name_max_len,
                name_min_len,
                loose_match=manager_loose_match,
            )
            if not expected and zh_en_map:
                expected = _map_expected_with_zh_map(
                    _get_expected_objects(item, name_max_len, name_min_len, smart_expand),
                    zh_en_map,
                    name_max_len,
                    name_min_len,
                    loose_match=zh_en_loose_match,
                    keep_unknown=zh_en_keep_unknown,
                )
            if not expected:
                per_sample.append({
                    "image_path": img_path,
                    "objects": 0,
                    "pos_ok": 0,
                    "mean_iou": 0.0,
                    "rel_total": 0,
                    "rel_matched": 0,
                    "rel_ok": 0,
                    "manager_pass": pass_to_dino,
                })
                continue

        if (not use_manager) and use_name_en:
            expected = _map_expected_with_name_en(expected)
            if not expected:
                continue
        if (not use_manager) and (not use_name_en) and zh_en_map:
            expected = _map_expected_with_zh_map(
                expected,
                zh_en_map,
                name_max_len,
                name_min_len,
                loose_match=zh_en_loose_match,
                keep_unknown=zh_en_keep_unknown,
            )
            if not expected:
                continue

        total_samples += 1
        total_objects += len(expected)

        v = verifier.verify_layout(
            image,
            expected,
            threshold=dino_thresh,
            iou_threshold=iou_thresh,
            match_label=dino_match_label,
            debug_topk=debug_dino_topk if debug_dino and debug_count < debug_dino else 0,
        )
        if debug_dino and debug_count < debug_dino:
            dbg = v.get("debug")
            if dbg:
                payload = {
                    "image_path": img_path,
                    "prompt": item.get("prompt", ""),
                    "expected_names": [e.get("name", "") for e in expected],
                    "text_prompt": dbg.get("text_prompt"),
                    "match_label": dbg.get("match_label"),
                    "topk": dbg.get("topk", []),
                }
                if debug_f:
                    debug_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                else:
                    print("[DINO-DEBUG]", json.dumps(payload, ensure_ascii=False))
                debug_count += 1
        details = v.get("details", []) or []

        # 对齐预测 bbox（按 expected 顺序）
        pred_boxes = [None] * len(expected)
        for idx, d in enumerate(details[:len(expected)]):
            iou = float(d.get("iou", 0.0))
            iou_sum += iou
            if iou >= iou_thresh:
                pos_ok += 1
            if d.get("detected_bbox") is None:
                pred_boxes[idx] = None
            else:
                # detected_bbox 是像素坐标，转为 0-1
                w, h = image.size
                pb = d["detected_bbox"]
                if isinstance(pb, (list, tuple)) and len(pb) == 4:
                    pred_boxes[idx] = [pb[0] / w, pb[1] / h, pb[2] / w, pb[3] / h]
                else:
                    pred_boxes[idx] = None

        detected_objects += sum(1 for b in pred_boxes if b is not None)
        dino_samples += 1

        # 关系评估（按 GT 关系）
        rel_pairs = 0
        rel_pairs_matched = 0
        rel_pairs_ok = 0
        for i in range(len(expected)):
            for j in range(i + 1, len(expected)):
                rel = _relation_from_gt(expected[i], expected[j], min_sep=rel_min_sep)
                if rel is None:
                    continue
                rel_pairs += 1
                if pred_boxes[i] is None or pred_boxes[j] is None:
                    continue
                rel_pairs_matched += 1
                if _relation_ok(rel, pred_boxes[i], pred_boxes[j], margin=rel_margin):
                    rel_pairs_ok += 1

        rel_total += rel_pairs
        rel_matched += rel_pairs_matched
        rel_ok += rel_pairs_ok

        per_sample.append({
            "image_path": img_path,
            "objects": len(expected),
            "pos_ok": sum(1 for d in details if float(d.get("iou", 0.0)) >= iou_thresh),
            "mean_iou": sum(float(d.get("iou", 0.0)) for d in details) / max(len(details), 1),
            "rel_total": rel_pairs,
            "rel_matched": rel_pairs_matched,
            "rel_ok": rel_pairs_ok,
            "manager_pass": True if use_manager else None,
        })

    if debug_f:
        debug_f.close()

    metrics = {
        "samples": total_samples,
        "objects_total": total_objects,
        "objects_detected": detected_objects,
        "pos_acc": float(pos_ok) / total_objects if total_objects else 0.0,
        "mean_iou": float(iou_sum) / total_objects if total_objects else 0.0,
        "rel_acc_all": float(rel_ok) / rel_total if rel_total else 0.0,
        "rel_acc_matched": float(rel_ok) / rel_matched if rel_matched else 0.0,
        "rel_pairs_total": rel_total,
        "rel_pairs_matched": rel_matched,
        "manager_total": manager_total,
        "manager_pass": manager_pass,
        "manager_pass_rate": float(manager_pass) / manager_total if manager_total else 0.0,
        "dino_samples": dino_samples,
    }

    if save_jsonl:
        out_path = save_jsonl
        if distributed and world_size > 1:
            out_path = f"{save_jsonl}.rank{rank}"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in per_sample:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if distributed and torch.distributed.is_initialized():
        device = torch.device("cuda", rank if torch.cuda.is_available() else 0) if torch.cuda.is_available() else torch.device("cpu")
        t = torch.tensor(
            [
                float(total_samples),
                float(total_objects),
                float(detected_objects),
                float(pos_ok),
                float(iou_sum),
                float(rel_total),
                float(rel_matched),
                float(rel_ok),
                float(manager_total),
                float(manager_pass),
                float(dino_samples),
            ],
            dtype=torch.float64,
            device=device,
        )
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        (total_samples,
         total_objects,
         detected_objects,
         pos_ok,
         iou_sum,
         rel_total,
         rel_matched,
         rel_ok,
         manager_total,
         manager_pass,
         dino_samples) = t.tolist()
        metrics = {
            "samples": int(total_samples),
            "objects_total": int(total_objects),
            "objects_detected": int(detected_objects),
            "pos_acc": float(pos_ok) / total_objects if total_objects else 0.0,
            "mean_iou": float(iou_sum) / total_objects if total_objects else 0.0,
            "rel_acc_all": float(rel_ok) / rel_total if rel_total else 0.0,
            "rel_acc_matched": float(rel_ok) / rel_matched if rel_matched else 0.0,
            "rel_pairs_total": int(rel_total),
            "rel_pairs_matched": int(rel_matched),
            "manager_total": int(manager_total),
            "manager_pass": int(manager_pass),
            "manager_pass_rate": float(manager_pass) / manager_total if manager_total else 0.0,
            "dino_samples": int(dino_samples),
        }

    return metrics


def _print_metrics(title: str, m: Dict):
    print(f"\n=== {title} ===")
    print(f"samples: {m['samples']}")
    print(f"objects_total: {m['objects_total']}")
    print(f"objects_detected: {m['objects_detected']}")
    if "manager_total" in m and m["manager_total"] > 0:
        print(f"manager_pass_rate: {m['manager_pass_rate']:.4f} ({m['manager_pass']}/{m['manager_total']})")
        print(f"dino_samples: {m.get('dino_samples', 0)}")
    print(f"pos_acc (IoU>=thr): {m['pos_acc']:.4f}")
    print(f"mean_iou: {m['mean_iou']:.4f}")
    print(f"rel_acc_all: {m['rel_acc_all']:.4f}")
    print(f"rel_acc_matched: {m['rel_acc_matched']:.4f}")
    print(f"rel_pairs_total: {m['rel_pairs_total']}")
    print(f"rel_pairs_matched: {m['rel_pairs_matched']}")


def main():
    parser = argparse.ArgumentParser(description="Small eval: bbox+phrase vs bbox-only")
    parser.add_argument("--results-phrase", type=str, help="JSONL from bbox+phrase generation")
    parser.add_argument("--results-bbox", type=str, help="JSONL from bbox-only generation")
    parser.add_argument("--results", type=str, help="Single JSONL to evaluate")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dino-model", type=str, default="/mnt/disk/lxh/models/grounding-dino-base")
    parser.add_argument("--dino-backend", choices=["auto", "hf", "official"], default="auto")
    parser.add_argument("--dino-config", type=str, default=None, help="Official GroundingDINO config .py path")
    parser.add_argument("--dino-ckpt", type=str, default=None, help="Official GroundingDINO checkpoint .pth path")
    parser.add_argument("--dino-device", type=str, default="cuda")
    parser.add_argument("--dino-threshold", type=float, default=0.35)
    parser.add_argument("--dino-text-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--dino-no-label-filter", action="store_true", help="Ignore label match when assigning DINO boxes")
    parser.add_argument("--debug-dino", type=int, default=0, help="Print/debug first N samples' DINO labels")
    parser.add_argument("--debug-dino-topk", type=int, default=10, help="TopK labels to show in DINO debug")
    parser.add_argument("--debug-dino-jsonl", type=str, default=None, help="Write DINO debug info to jsonl")
    parser.add_argument("--rel-min-sep", type=float, default=0.1)
    parser.add_argument("--rel-margin", type=float, default=0.02)
    parser.add_argument("--adapter-scale", type=float, default=None, help="(no-op) adapter scale used during generation")
    parser.add_argument("--force-gate", action="store_true", help="(no-op) gate override used during generation")
    parser.add_argument("--gate-value", type=float, default=None, help="(no-op) gate value used during generation")
    parser.add_argument("--use-qwen-manager", action="store_true", help="Use Qwen2-VL manager to map zh->en and gate DINO")
    parser.add_argument("--manager-no-gate", action="store_true", help="Do not gate samples by manager pass_to_dino")
    parser.add_argument("--manager-loose-match", action="store_true", help="Loosely match manager zh names to expected")
    parser.add_argument("--zh-en-map", type=str, default=None, help="JSON dict mapping zh object names to English")
    parser.add_argument("--zh-en-loose-match", action="store_true", help="Loosely match zh-en map keys")
    parser.add_argument("--zh-en-keep-unknown", action="store_true", help="Keep unmapped zh names instead of skipping")
    parser.add_argument("--use-name-en", action="store_true", help="Use name_en field for DINO phrases (when available)")
    parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--qwen-device", type=str, default="cuda")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=256)
    parser.add_argument("--manager-log", type=str, default="outputs/manager_logs.jsonl", help="Path to save manager JSONL logs")
    parser.add_argument("--name-max-len", type=int, default=10)
    parser.add_argument("--name-min-len", type=int, default=1)
    parser.add_argument("--smart-expand", action="store_true", help="Apply smart bbox expansion for tiny objects")
    parser.add_argument("--save-phrase-jsonl", type=str, default=None)
    parser.add_argument("--save-bbox-jsonl", type=str, default=None)
    parser.add_argument("--save-jsonl", type=str, default=None)
    args = parser.parse_args()
    if args.adapter_scale is not None or args.force_gate or args.gate_value is not None:
        print(f"[info] eval-only flags: adapter_scale={args.adapter_scale}, force_gate={args.force_gate}, gate_value={args.gate_value}")

    zh_en_map = _load_zh_en_map(args.zh_en_map)

    distributed, rank, world_size, local_rank = _dist_info()
    if distributed:
        if args.dino_device.startswith("cuda"):
            args.dino_device = f"cuda:{local_rank}"
        if args.qwen_device.startswith("cuda"):
            args.qwen_device = f"cuda:{local_rank}"
        if args.manager_log:
            args.manager_log = f"{args.manager_log}.rank{local_rank}"

    verifier = GroundingDinoVerifier(
        model_id=args.dino_model,
        device=args.dino_device,
        backend=args.dino_backend,
        config_path=args.dino_config,
        checkpoint_path=args.dino_ckpt,
        box_threshold=args.dino_threshold,
        text_threshold=args.dino_text_threshold,
    )
    manager = None
    if args.use_qwen_manager:
        manager = QwenSemanticVerifier(model_path=args.qwen_model, device=args.qwen_device)

    if args.results:
        debug_path = args.debug_dino_jsonl
        if debug_path and distributed and world_size > 1:
            debug_path = f"{debug_path}.rank{local_rank}"
        m = evaluate_results(
            args.results,
            verifier,
            manager,
            args.use_qwen_manager,
            args.manager_no_gate,
            args.manager_loose_match,
            zh_en_map,
            args.zh_en_loose_match,
            args.zh_en_keep_unknown,
            args.use_name_en,
            (not args.dino_no_label_filter),
            args.debug_dino,
            args.debug_dino_topk,
            debug_path,
            args.qwen_max_new_tokens,
            args.manager_log,
            args.max_samples,
            args.seed,
            args.dino_threshold,
            args.iou_threshold,
            args.rel_min_sep,
            args.rel_margin,
            args.name_max_len,
            args.name_min_len,
            distributed,
            rank,
            world_size,
            args.smart_expand,
            save_jsonl=args.save_jsonl,
        )
        if (not distributed) or rank == 0:
            _print_metrics("Eval", m)
        return

    if not args.results_phrase or not args.results_bbox:
        print("请提供 --results-phrase 和 --results-bbox，或使用 --results 评估单个文件。")
        return

    debug_phrase = args.debug_dino_jsonl
    debug_bbox = args.debug_dino_jsonl
    if args.debug_dino_jsonl:
        debug_phrase = f"{args.debug_dino_jsonl}.phrase"
        debug_bbox = f"{args.debug_dino_jsonl}.bbox"
    if distributed and world_size > 1 and args.debug_dino_jsonl:
        debug_phrase = f"{debug_phrase}.rank{local_rank}"
        debug_bbox = f"{debug_bbox}.rank{local_rank}"

    m_phrase = evaluate_results(
        args.results_phrase,
        verifier,
        manager,
        args.use_qwen_manager,
        args.manager_no_gate,
        args.manager_loose_match,
        zh_en_map,
        args.zh_en_loose_match,
        args.zh_en_keep_unknown,
        args.use_name_en,
        (not args.dino_no_label_filter),
        args.debug_dino,
        args.debug_dino_topk,
        debug_phrase,
        args.qwen_max_new_tokens,
        args.manager_log,
        args.max_samples,
        args.seed,
        args.dino_threshold,
        args.iou_threshold,
        args.rel_min_sep,
        args.rel_margin,
        args.name_max_len,
        args.name_min_len,
        distributed,
        rank,
        world_size,
        args.smart_expand,
        save_jsonl=args.save_phrase_jsonl,
    )
    m_bbox = evaluate_results(
        args.results_bbox,
        verifier,
        manager,
        args.use_qwen_manager,
        args.manager_no_gate,
        args.manager_loose_match,
        zh_en_map,
        args.zh_en_loose_match,
        args.zh_en_keep_unknown,
        args.use_name_en,
        (not args.dino_no_label_filter),
        args.debug_dino,
        args.debug_dino_topk,
        debug_bbox,
        args.qwen_max_new_tokens,
        args.manager_log,
        args.max_samples,
        args.seed,
        args.dino_threshold,
        args.iou_threshold,
        args.rel_min_sep,
        args.rel_margin,
        args.name_max_len,
        args.name_min_len,
        distributed,
        rank,
        world_size,
        args.smart_expand,
        save_jsonl=args.save_bbox_jsonl,
    )

    if (not distributed) or rank == 0:
        _print_metrics("bbox+phrase", m_phrase)
        _print_metrics("bbox-only", m_bbox)

    if distributed and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
