"""Shared taxonomy for evaluation-only LIBERO failure analysis."""

from __future__ import annotations


FAILURE_TAXONOMY = (
    "目标选择/任务理解错误",
    "接近/位姿对齐失败",
    "抓取/接触失败",
    "搬运/放置失败",
    "机构交互失败",
    "动作结果判断错误",
    "偏差累积/恢复失败",
    "纯阶段规划错误",
    "其他/无法判断",
)

UNDETERMINED_STAGE = "无法判断"
MANUAL_REVIEW_EVIDENCE_MARKERS = ("无法判断", "遮挡", "未能确认", "可能")
