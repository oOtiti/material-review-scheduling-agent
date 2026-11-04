import re


GREETING = re.compile(r"(?:你好|您好|哈喽|嗨|在吗|早上好|晚上好|hi|hello|hey)[啊呀哦]?")
CONFIRMATION = re.compile(r"(?:确认|选)\s*(?:第|选项)?\s*[0-9一二三四五六七八九十]+\s*[号项]?")
SCHEDULE_WORD = re.compile(r"审核|排(?:个)?期|档期|改期")
PUBLISH_REQUEST = re.compile(r"(?:请|帮我|我要|我想)?发布(?:这个|这条)?(?:素材|视频|短视频)")
AMBIGUOUS_SCHEDULE = re.compile(r"(?:(?:能|可以|麻烦)?帮我|我想|我要)?"
                               r"(?:排(?:个)?期|安排档期|安排排期|改期)(?:吗|一下)?")


def respond_to_message(message):
    text = message.strip(" \t\r\n，。！？、…,.!?~～").casefold()
    if not text:
        return {"status": "empty_input", "response": "请告诉我你想审核哪个素材或安排什么档期。"}
    if GREETING.fullmatch(text):
        return {"status": "greeting", "response": "你好！我可以帮你做素材审核和发布排期。"}

    text = re.sub(r"^(?:你好|您好|hi|hello)[，,\s]+", "", text, count=1)
    if text in {"谢谢", "谢谢你", "多谢", "thanks", "thank you"}:
        return {"status": "thanks", "response": "不客气，后续有素材审核或发布排期需求可以继续。"}
    if text in {"帮助", "怎么用", "你能做什么", "你会做什么", "help"}:
        return {"status": "scope_help", "response": "我可以审核素材、查询发布档期和准备草稿。请用 start 提交排期请求。"}
    if CONFIRMATION.fullmatch(text):
        return {"status": "confirmation_required",
                "response": "请使用 confirm，并提供当前提案的 request_id、proposal_id 和选项编号。"}
    material_request = any(word in text for word in ("素材", "视频", "内容发布")) and SCHEDULE_WORD.search(text)
    if material_request or PUBLISH_REQUEST.fullmatch(text):
        return {"status": "needs_input",
                "response": "请提供素材编号、发布资源、首选时段、时长和允许的时间窗口，再用 start 提交请求。",
                "example_input": "examples/normal.input.json"}
    if AMBIGUOUS_SCHEDULE.fullmatch(text):
        return {"status": "scope_clarification", "response": "你要安排的是内容素材的发布档期吗？"}
    return {"status": "out_of_scope",
            "response": "这项需求超出我的处理范围。我可以帮你做素材审核、查询发布排期和准备草稿。"}
