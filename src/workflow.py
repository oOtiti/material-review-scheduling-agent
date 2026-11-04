import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone


CHINA = timezone(timedelta(hours=8))
MAX_STEPS = 8
MAX_TOOL_CALLS = 12
CASE_TOP_K = 3


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Time values must be ISO 8601 strings with an offset.")
    time = datetime.fromisoformat(value)
    if time.utcoffset() is None:
        raise ValueError("Time values must include a UTC offset.")
    return time.astimezone(CHINA)


def slot_at(start, minutes):
    return {"start": start.isoformat(), "end": (start + timedelta(minutes=minutes)).isoformat()}


def digest(value):
    body = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()[:24]


class Workflow:
    def __init__(self, store, data, model, *, context_enabled=True):
        if not isinstance(data, dict) or any(not isinstance(data.get(key), expected)
                                            for key, expected in (("version", str), ("rule", dict),
                                                                  ("materials", dict), ("bookings", list))):
            raise ValueError("Business data requires a version, rule, materials and bookings.")
        self.store, self.data, self.model = store, data, model
        self.context_enabled = context_enabled

    def start(self, request):
        if not isinstance(request, dict) or not isinstance(request.get("request_id"), str):
            raise ValueError("An object with a string request_id is required.")
        request_id = request["request_id"].strip()
        if not request_id:
            raise ValueError("request_id must be nonempty.")
        request = dict(request, request_id=request_id)
        with self.store.transaction() as conn:
            prior = self.store.session(conn, request_id)
        if prior and prior["result"]["status"] not in {"needs_input", "needs_reassessment", "incomplete"}:
            if prior["request"] != request:
                return {"request_id": request_id, "status": "request_conflict"}
            return self.status(request_id)
        revision = prior["revision"] + 1 if prior else 1
        result = self.assess(request, revision)
        with self.store.transaction() as conn:
            current = self.store.session(conn, request_id)
            if current != prior:
                return {"request_id": request_id, "status": "request_changed"}
            self.store.put(conn, {"request": request, "revision": revision, "result": result})
        return result

    def assess(self, request, revision):
        result = {
            "request_id": request["request_id"], "status": "assessing", "revision": revision,
            "mode": self.model.label, "data_version": self.data["version"],
            "plan_route": "original_slot", "replans": [],
            "model_usage": {"decision_calls": 0, "reported_calls": 0, "input_tokens": 0, "output_tokens": 0},
            "plan": [
                {"goal": "verify_material", "sources": ["sop:material-review", "request:material_id"]},
                {"goal": "assess_schedule", "sources": ["sop:schedule", "request:time_window"],
                 "requires": ["verify_material"]},
                {"goal": "request_confirmation", "sources": ["sop:approval", "tool:verified_slots"],
                 "requires": ["assess_schedule"]},
                {"goal": "save_draft", "sources": ["sop:draft-write", "event:confirmed_proposal"],
                 "requires": ["request_confirmation"]},
            ],
            "trace": [],
        }
        missing = [
            key for key in ("material_id", "resource", "preferred_start", "slot_minutes", "window_start", "window_end")
            if request.get(key) is None or request.get(key) == ""
        ]
        if missing:
            return dict(result, status="needs_input", reason="MISSING_REQUEST_FIELDS", missing=missing)
        self.validate_request(request)
        preference = self.preference(request)
        seen = {}
        rejected = {}
        repeated = Counter()
        stalled = tools_used = 0
        for step in range(1, MAX_STEPS + 1):
            context = self.context(request, seen, rejected, result["plan_route"], preference)
            result["model_usage"]["decision_calls"] += 1
            try:
                decision = self.model.decide(context)
            except Exception as exc:
                return dict(result, status="incomplete", reason="MODEL_ERROR", error_type=type(exc).__name__)
            if isinstance(decision, dict):
                self.record_token_usage(result["model_usage"], decision.get("usage"))
            action = decision.get("action") if isinstance(decision, dict) else None
            basis = decision.get("basis") if isinstance(decision, dict) else None
            if action not in context["allowed"] or not isinstance(basis, str) or not basis.strip():
                return dict(result, status="incomplete", reason="INVALID_ACTION")
            if action in {"ask_input", "finish_blocked", "propose_original", "propose_alternatives"}:
                self.finish(result, request, seen, action)
                args = {"request_id": request["request_id"]}
                if result.get("reason") == "UNVERIFIED_PROPOSAL":
                    rejected[action] = result.pop("reason")
                    result["status"] = "assessing"
                    observation = {"status": "rejected", "reason": rejected[action], "slot": context["slot"]}
                    self.trace(result, step, action, args, observation, basis, "end", context["sources"])
                    continue
                observation = {"status": result["status"], "reason": result.get("reason"),
                               "candidate_count": len(result.get("candidates", []))}
                self.trace(result, step, action, args, observation, basis, "end", context["sources"])
                return result
            if repeated[action] >= 2 or stalled >= 3:
                return dict(result, status="incomplete", reason="NO_PROGRESS")
            if tools_used >= MAX_TOOL_CALLS:
                return dict(result, status="incomplete", reason="TOOL_LIMIT")
            try:
                args, observation = self.read(action, request, preference)
            except (KeyError, TypeError, ValueError) as exc:
                return dict(result, status="incomplete", reason="TOOL_ERROR",
                            failed_tool=action, error_type=type(exc).__name__)
            tools_used += 1
            if action not in seen or seen[action] != observation:
                stalled = 0
                repeated.clear()
            else:
                stalled += 1
            repeated[action] += 1
            seen[action] = observation
            self.trace(result, step, action, args, observation, basis, "end", context["sources"])
            self.update_plan(result, request, action, observation, step)
        return dict(result, status="incomplete", reason="STEP_LIMIT")

    @staticmethod
    def record_token_usage(totals, usage):
        if not isinstance(usage, dict):
            return
        token_fields = ("input_tokens", "output_tokens")
        if all(type(usage.get(key)) is int and usage[key] >= 0 for key in token_fields):
            totals["reported_calls"] += 1
            for key in token_fields:
                totals[key] += usage[key]

    def update_plan(self, result, request, action, observation, step):
        status = observation.get("status")
        if (action, status) == ("query_slot", "occupied"):
            if request.get("flexible") is None:
                route, reason, goals = "clarify", "FLEXIBILITY_UNSPECIFIED", ["ask_input"]
            elif request["flexible"]:
                goals = ["find_slots", "propose_alternatives", "request_confirmation"]
                if self.context_enabled:
                    goals.insert(0, "recall_cases")
                route, reason = "alternative_slots", "PREFERRED_SLOT_OCCUPIED"
            else:
                route, reason, goals = "blocked", "LOCKED_SLOT_OCCUPIED", ["finish_blocked"]
        elif (action, status) == ("get_material", "missing"):
            route, reason, goals = "clarify", "MATERIAL_FIELDS_MISSING", ["ask_input"]
        elif (action, status) == ("review_material", "blocked"):
            route, reason, goals = "blocked", "AUDIT_BLOCKED", ["finish_blocked"]
        elif (action, status) == ("find_slots", "none"):
            route, reason, goals = "blocked", "NO_ELIGIBLE_SLOT", ["finish_blocked"]
        else:
            return
        self.replan(result, route, reason, action, status, step, goals)

    @staticmethod
    def replan(result, route, reason, action, status, step, goals):
        previous = result.get("plan_route", "original_slot")
        if previous != route:
            event = {"from": previous, "to": route, "reason": reason,
                     "trigger": {"step": step, "action": action, "status": status},
                     "remaining_goals": goals}
            result["replans"] = [*result.get("replans", []), event]
            result["plan_route"] = route

    @staticmethod
    def validate_request(request):
        for key in ("material_id", "resource"):
            if not isinstance(request[key], str) or not request[key].strip():
                raise ValueError(f"{key} must be a nonempty string.")
        if type(request["slot_minutes"]) is not int or not 5 <= request["slot_minutes"] <= 240:
            raise ValueError("slot_minutes must be an integer from 5 to 240.")
        if request.get("flexible") is not None and type(request["flexible"]) is not bool:
            raise ValueError("flexible must be a boolean.")
        start, preferred, end = (timestamp(request[key]) for key in ("window_start", "preferred_start", "window_end"))
        if not start <= preferred or preferred + timedelta(minutes=request["slot_minutes"]) > end:
            raise ValueError("The preferred slot must fit in the allowed window.")
        if end - start > timedelta(days=1):
            raise ValueError("The demo supports a scheduling window of up to one day.")

    def context(self, request, seen, rejected, plan_route, preference):
        material = seen.get("get_material", {}).get("status", "unchecked")
        review = seen.get("review_material", {}).get("status", "unchecked")
        slot = seen.get("query_slot", {}).get("status", "unchecked")
        alternatives = seen.get("find_slots", {}).get("status", "unchecked")
        flexible = request.get("flexible")
        allowed = ["get_material"]
        if plan_route == "clarify":
            allowed.append("ask_input")
        if material == "present":
            allowed.append("review_material")
        if review == "blocked":
            allowed.append("finish_blocked")
        if review == "passed":
            allowed.append("query_slot")
        recall_pending = self.context_enabled and plan_route == "alternative_slots" and "recall_cases" not in seen
        if recall_pending:
            allowed.append("recall_cases")
        if slot != "unchecked" and plan_route in {"original_slot", "alternative_slots"}:
            allowed.append("propose_original")
        if plan_route == "alternative_slots" and not recall_pending:
            allowed.append("find_slots")
        if alternatives == "found":
            allowed.append("propose_alternatives")
        if (slot == "occupied" and flexible is False) or alternatives == "none":
            allowed.append("finish_blocked")
        cards = []
        sources = [{"mode": "observation", "ref": f"tool:{action}"} for action in seen]
        if self.context_enabled and material == "present" and review == "unchecked":
            rule = self.data["rule"]
            cards.append({"mode": "sop", "ref": f"rule:{rule['id']}@{rule['version']}",
                          "rule": "Rights must match the current allowed_rights", "allowed_rights": rule["allowed_rights"]})
        if self.context_enabled and review == "passed":
            cards.append({"mode": "sop", "ref": "sop:schedule-window",
                          "start": request["window_start"], "end": request["window_end"]})
        profile, order, source = preference
        if slot == "occupied" and flexible and order:
            cards.append({"mode": "sop", "ref": f"preference:{profile}",
                          "latest_first": order.get("latest_first", False), "source": source})
        cases = seen.get("recall_cases", {}).get("cases", [])
        sources += [{"mode": "sop", "ref": card["ref"]} for card in cards]
        sources += [{"mode": "retrieval", "ref": f"case:{case['id']}"} for case in cases]
        sources += [{"mode": "guard", "ref": f"rejected:{action}"} for action in rejected]
        return {"allowed": [action for action in dict.fromkeys(allowed) if action not in rejected],
                "plan_route": plan_route, "material": material, "review": review,
                "slot": slot, "alternatives": alternatives, "flexible": flexible,
                "observed": list(seen), "case_advice": cases, "sop_context": cards, "sources": sources,
                "guard_feedback": rejected}

    def material(self, material_id):
        material = self.data["materials"].get(material_id, {})
        missing = [key for key in ("title", "rights", "revision") if not material.get(key)]
        return {"status": "missing", "missing": missing} if missing else {
            "status": "present", "material": material,
        }

    def audit(self, material_id):
        material = self.material(material_id)
        rule = self.data["rule"]
        if material["status"] == "missing":
            return material
        facts = material["material"]
        passed = facts["rights"] == rule["allowed_rights"]
        return {"status": "passed" if passed else "blocked", "rule_id": rule["id"],
                "rule_version": rule["version"], "field": "rights", "observed": facts["rights"],
                "expected": rule["allowed_rights"], "material_revision": facts["revision"],
                "evidence_version": digest([rule, facts])}

    def available(self, resource, slot):
        start, end = timestamp(slot["start"]), timestamp(slot["end"])
        occupied = False
        for row in self.data["bookings"]:
            if row["resource"] != resource:
                continue
            booking_start, booking_end = timestamp(row["start"]), timestamp(row["end"])
            if booking_start >= booking_end:
                raise ValueError("A booking must have a positive duration.")
            occupied |= start < booking_end and booking_start < end
        return not occupied

    def read(self, action, request, preference):
        material_id, resource = request["material_id"], request["resource"]
        if action == "get_material":
            return {"material_id": material_id}, self.material(material_id)
        if action == "review_material":
            return {"material_id": material_id}, self.audit(material_id)
        if action == "query_slot":
            slot = slot_at(timestamp(request["preferred_start"]), request["slot_minutes"])
            return {"resource": resource, **slot}, {
                "status": "free" if self.available(resource, slot) else "occupied", "slot": slot,
            }
        if action == "recall_cases":
            return self.recall_cases(request)
        if action != "find_slots":
            raise ValueError("Unknown read tool.")
        args = {key: request[key] for key in ("resource", "window_start", "window_end", "slot_minutes")}
        args["flexible"] = request.get("flexible") is True
        start, end = timestamp(request["window_start"]), timestamp(request["window_end"])
        minutes = request["slot_minutes"]
        duration = timedelta(minutes=minutes)
        candidates = []
        if args["flexible"]:
            starts = {timestamp(row["end"]) for row in self.data["bookings"] if row["resource"] == resource}
            grid_start = start
            while grid_start + duration <= end:
                starts.add(grid_start)
                grid_start += duration
            for candidate_start in sorted(starts):
                if start <= candidate_start and candidate_start + duration <= end:
                    slot = slot_at(candidate_start, minutes)
                    if self.available(resource, slot):
                        candidates.append(slot)
        _, order, source = preference
        if order.get("latest_first"):
            candidates.reverse()
        return args, {"status": "found" if candidates else "none", "candidates": candidates[:3],
                      "ranked_with_preference": bool(order), "preference_source": source}

    def recall_cases(self, request):
        filters = {"topic": "occupied", "rule_version": self.data["rule"]["version"]}
        weights = {"resource": 2, "slot_minutes": 1}
        hits = []
        for case in self.data.get("cases", []):
            if any(case.get(key) != value for key, value in filters.items()):
                continue
            matched_on = [key for key in weights if case.get(key) == request[key]]
            hits.append({"id": case["id"], "note": case["note"][:400], "rule_version": case["rule_version"],
                         "matched_on": matched_on, "score": sum(weights[key] for key in matched_on)})
        hits.sort(key=lambda hit: (-hit["score"], hit["id"]))
        args = dict(filters, resource=request["resource"], slot_minutes=request["slot_minutes"], top_k=CASE_TOP_K)
        return args, {"eligible_count": len(hits), "cases": hits[:CASE_TOP_K]}

    def preference(self, request):
        profile = request.get("profile")
        if self.context_enabled and isinstance(profile, str):
            saved = self.store.preference(profile)
            if saved:
                return profile, saved, "saved"
            default = self.data.get("preferences", {}).get(profile, {})
            return profile, default, "fixture" if default else None
        return None, {}, None

    @staticmethod
    def trace(result, step, action, args, observation, basis, next_action, sources=()):
        if result["trace"]:
            result["trace"][-1] = dict(result["trace"][-1], next_action=action)
        result["trace"].append({
            "step": step, "action": action, "args": args, "observation": observation,
            "decision_basis": basis.strip()[:240], "next_action": next_action,
            "context_sources": list(sources), "plan_route": result.get("plan_route", "original_slot"),
        })

    @staticmethod
    def finish(result, request, seen, action):
        audit = seen.get("review_material", {})
        slot = seen.get("query_slot", {})
        if action == "ask_input":
            if slot.get("status") == "occupied" and request.get("flexible") is None:
                result.update(status="needs_input", reason="FLEXIBILITY_UNSPECIFIED", missing=["flexible"])
            else:
                result.update(status="needs_input", reason="MISSING_MATERIAL_FIELDS",
                              missing=seen["get_material"]["missing"])
        elif action == "finish_blocked":
            result.update(status="blocked", reason="REVIEW_BLOCKED" if audit["status"] == "blocked"
                          else "NO_ELIGIBLE_SLOT", review=audit)
        else:
            candidates = ([slot["slot"]] if action == "propose_original" and slot["status"] == "free"
                          else seen.get("find_slots", {}).get("candidates", [])
                          if action == "propose_alternatives" else [])
            if audit.get("status") != "passed" or not candidates:
                result.update(status="incomplete", reason="UNVERIFIED_PROPOSAL")
                return
            proposal_id = digest([request, result["revision"], audit, candidates])
            result.update(status="awaiting_confirmation", proposal_id=proposal_id, review=audit,
                          requested_slot=slot, candidates=[dict(s, choice=i) for i, s in enumerate(candidates, 1)])

    def confirm(self, request_id, proposal_id, choice, fault=None):
        with self.store.transaction() as conn:
            session = self.store.session(conn, request_id)
            previous = self.store.draft(conn, request_id)
            if (previous and type(choice) is int
                    and previous["proposal_id"] == proposal_id and previous["choice"] == choice):
                return dict(previous, duplicate=True)
            result = session["result"] if session else {}
            candidates = result.get("candidates", [])
            selected = next((slot for slot in candidates if slot["choice"] == choice), None)
            if (previous or result.get("status") not in {"awaiting_confirmation", "save_failed"}
                    or proposal_id != result.get("proposal_id") or selected is None or type(choice) is not int):
                return {"request_id": request_id, "status": "invalid_confirmation"}
            request = session["request"]
            try:
                current_audit = self.audit(request["material_id"])
                currently_free = self.available(request["resource"], selected)
            except (KeyError, TypeError, ValueError):
                deferred = dict(result, reason="TOOL_ERROR")
                return self.record(conn, session, self.save_trace(deferred, proposal_id, choice))
            if current_audit != result["review"] or not currently_free:
                stale = dict(result, status="needs_reassessment", reason="STALE_PROPOSAL")
                stale = self.save_trace(stale, proposal_id, choice)
                self.replan(stale, "reassess", "STALE_PROPOSAL", "save_confirmed_draft",
                            stale["status"], len(stale["trace"]), ["verify_material", "assess_schedule"])
                return self.record(conn, session, stale)

            saved = dict(result, status="draft_saved", draft_id=digest([request_id, proposal_id, choice]),
                         choice=choice, saved_slot=selected, duplicate=False)
            saved.pop("reason", None)
            saved = self.save_trace(saved, proposal_id, choice)
            conn.execute("SAVEPOINT draft_write")
            self.store.insert_draft(conn, request_id, saved)
            if fault == "save_failure":
                conn.execute("ROLLBACK TO draft_write")
                conn.execute("RELEASE draft_write")
                failed = dict(result, status="save_failed", reason="SAVE_FAILED")
                return self.record(conn, session, self.save_trace(failed, proposal_id, choice))
            conn.execute("RELEASE draft_write")
            if fault == "lost_response":
                unknown = dict(result, status="save_unknown", reason="RESPONSE_LOST")
                return self.record(conn, session, self.save_trace(unknown, proposal_id, choice))
            return self.record(conn, session, saved)

    def save_trace(self, result, proposal_id, choice):
        result = dict(result, trace=list(result["trace"]))
        self.trace(
            result, len(result["trace"]) + 1, "save_confirmed_draft",
            {"request_id": result["request_id"], "proposal_id": proposal_id, "choice": choice,
             "authority": "explicit_user_confirmation"},
            {"status": result["status"], "draft_id": result.get("draft_id"), "reason": result.get("reason")},
            "状态机核验了用户确认，按实际保存结果返回。", "end",
        )
        return result

    def record(self, conn, session, result):
        self.store.put(conn, dict(session, result=result))
        return result

    def status(self, request_id):
        with self.store.transaction() as conn:
            session = self.store.session(conn, request_id)
            if session is None:
                return {"request_id": request_id, "status": "not_found"}
            result = session["result"]
            if result["status"] == "save_unknown":
                draft = self.store.draft(conn, request_id)
                if draft:
                    return self.record(conn, session, draft)
            return result
