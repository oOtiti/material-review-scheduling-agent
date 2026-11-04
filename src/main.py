import argparse
import json
import sqlite3
import sys
from pathlib import Path

from .intake import respond_to_message
from .model import LangChainModel, OfflineModel
from .storage import Store
from .workflow import Workflow


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def with_response(result):
    status = result["status"]
    if status == "awaiting_confirmation":
        candidates = "、".join(
            f"{slot['choice']}（{slot['start']} 至 {slot['end']}）" for slot in result["candidates"]
        )
        occupied = result["requested_slot"]["status"] == "occupied"
        slot = result["requested_slot"]["slot"]
        target = f"原时段 {slot['start']} 至 {slot['end']} 已占用。" if occupied else "原时段可用。"
        review = result["review"]
        if result.get("reason") == "TOOL_ERROR":
            response = "确认时暂时无法复查审核或档期，提案已保留。请在数据恢复后重新确认。"
        else:
            response = (f"素材通过 {review['rule_id']}@{review['rule_version']} 审核。{target}"
                        f"允许窗口内可选 {candidates}。请确认提案 {result['proposal_id']} 的时段编号。")
    elif status == "needs_input":
        if result.get("reason") == "FLEXIBILITY_UNSPECIFIED":
            response = "原时段已被占用。是否允许在你给定的窗口内改期？请用同一个 request_id 补充 flexible（true 或 false）。"
        else:
            response = f"请补充 {', '.join(result['missing'])} 后重新评估。"
    elif status == "blocked":
        review = result["review"]
        response = (f"规则 {review['rule_id']}@{review['rule_version']} 拦截。"
                    f"{review['field']} 为 {review['observed']}，规则要求 {review['expected']}。"
                    if result["reason"] == "REVIEW_BLOCKED" else "当前允许的时间范围内没有可用时段，请调整时间要求。")
    elif status == "draft_saved":
        slot = result["saved_slot"]
        response = f"草稿 {result['draft_id']} 已保存，时段为 {slot['start']} 至 {slot['end']}。"
    else:
        response = {
            "save_failed": "保存失败，原提案仍可重新确认后重试。",
            "save_unknown": "保存结果待核实，请使用 status 查询原请求。",
            "needs_reassessment": "审核或档期已变化，请用原请求重新评估并确认。",
            "incomplete": f"自动执行暂未完成（{result.get('reason')}），可以重新评估。",
            "invalid_confirmation": "确认与当前提案不匹配，请核对提案和时段编号。",
            "request_conflict": "请求编号已用于其他内容，请使用新编号。",
            "request_changed": "评估期间请求发生变化，请查询当前状态。",
            "not_found": "未找到该请求。",
        }.get(status, status)
    return dict(result, response=response)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Review material and propose a publication schedule draft.")
    parser.add_argument("--db", default="workflow.db")
    parser.add_argument("--data", default=ROOT / "data/demo.json")
    parser.add_argument("--mode", choices=["offline", "langchain"], default="offline")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("chat", help="Route a short message to help or clarification.").add_argument("message")
    commands.add_parser("start").add_argument("input")
    confirm = commands.add_parser("confirm")
    confirm.add_argument("request_id")
    confirm.add_argument("proposal_id")
    confirm.add_argument("choice", type=int)
    commands.add_parser("status").add_argument("request_id")
    remember = commands.add_parser("remember-preference")
    remember.add_argument("profile")
    remember.add_argument("order", choices=["earliest", "latest"])
    args = parser.parse_args()
    if args.command == "chat":
        print(json.dumps(respond_to_message(args.message), ensure_ascii=False, indent=2))
        return 0
    try:
        store = Store(args.db)
        if args.command == "remember-preference":
            profile = args.profile.strip()
            if not profile:
                raise ValueError("profile must be nonempty.")
            store.remember_preference(profile, latest_first=args.order == "latest")
            print(json.dumps({"status": "preference_saved", "profile": profile, "order": args.order},
                             ensure_ascii=False))
            return 0
        model = LangChainModel() if args.command == "start" and args.mode == "langchain" else OfflineModel()
        flow = Workflow(store, read_json(args.data), model)
        if args.command == "start":
            result = flow.start(read_json(args.input))
        elif args.command == "confirm":
            result = flow.confirm(args.request_id, args.proposal_id, args.choice)
        else:
            result = flow.status(args.request_id)
    except (OSError, ValueError, sqlite3.Error, ImportError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True), file=sys.stderr)
        return 2
    print(json.dumps(with_response(result), ensure_ascii=False, indent=2))
    return int(result.get("reason") == "TOOL_ERROR" or result["status"] in {
        "incomplete", "invalid_confirmation", "request_conflict",
        "request_changed", "not_found", "save_failed", "save_unknown",
    })


if __name__ == "__main__":
    raise SystemExit(main())
