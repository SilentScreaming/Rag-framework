"""Shared strict entity-list grammar; reference-compatibility scoring stays separate."""
from __future__ import annotations
import re
from rag_benchmark.schemas import AnswerBundleRecord, ListAnswerItemRecord
from rag_benchmark.stage_cache import canonical_hash

LIST_FORMAT_PROTOCOL = "entity-list-v1"


def is_explicit_abstention(answer: str) -> bool:
    """Recognize only exact protocol markers; semantic behavior is assessed independently."""
    return answer.strip().casefold().rstrip('.') in {"unknown", "i don't know", "i do not know"}


def strip_citation_markers(text):
    return re.sub(r"\[[^\[\]]+\]", " ", text)


def parse_entity_list(answer, citations=()):
    if not answer.strip() or is_explicit_abstention(answer):
        raise ValueError("empty_or_abstained_list")
    if any(c in answer for c in "\n\r;"):
        raise ValueError("ambiguous_list_format")
    pattern = re.compile(r'\s*("(?:[^"\n]|"")*"(?:\s*\[[^\[\]]+\])*|[^,"\n]+)\s*(,|$)')
    items, pos = [], 0
    while pos < len(answer):
        match = pattern.match(answer, pos)
        if not match or match.end() == pos:
            raise ValueError("ambiguous_list_format")
        raw = match.group(1)
        text = strip_citation_markers(raw).strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].replace('""', '"')
        if not re.search(r"\w", text) or len(text.split()) > 16 or text.endswith(("?", "!")):
            raise ValueError("not_an_entity_list")
        items.append(ListAnswerItemRecord(item_id=f"i{len(items)+1}", text=text,
            source_span=(match.start(1), match.end(1)), citation_ids=re.findall(r"\[([^\[\]]+)\]", raw),
            metadata={"source_text": raw, "source_answer_hash": canonical_hash(answer),
                      "format_protocol": LIST_FORMAT_PROTOCOL}))
        pos = match.end()
        if match.group(2) and not answer[pos:].strip():
            raise ValueError("trailing_empty_item")
    return items


def render_entity_list(items, *, include_citations=True):
    rendered = []
    for item in items:
        text = item.text
        if any(c in text for c in ',"'):
            text = '"' + text.replace('"', '""') + '"'
        rendered.append(text + (''.join(f' [{cid}]' for cid in item.citation_ids) if include_citations else ''))
    return ', '.join(rendered)


def validate_list_items(items):
    parsed = [x if isinstance(x, ListAnswerItemRecord) else ListAnswerItemRecord.model_validate(x) for x in items]
    if not parsed or len({i.item_id for i in parsed}) != len(parsed):
        raise ValueError('empty_or_duplicate_structured_list')
    for item in parsed:
        if not item.item_id.strip() or not re.search(r'\w', item.text) or len(item.text.split()) > 16:
            raise ValueError('not_an_entity_list')
        if any(c in item.text for c in '\n\r;') or item.text.endswith(('?', '!')):
            raise ValueError('not_an_entity_list')
    return parsed


def list_payload_metadata(answer, items):
    return {'list_format_status': 'computed', 'format_protocol': LIST_FORMAT_PROTOCOL,
            'source_answer_hash': canonical_hash(answer),
            'list_items_hash': canonical_hash([i.to_json_dict() for i in items])}


def list_items_for_prediction(prediction):
    """Use the structured answer list when available, otherwise parse its text."""
    bundle = prediction.answer_bundle
    if bundle is not None and bundle.list_items is not None:
        items = validate_list_items(bundle.list_items)
        current_hash = canonical_hash(prediction.answer)
        source_hash = bundle.metadata.get('source_answer_hash')
        items_hash = bundle.metadata.get('list_items_hash')
        actual_items_hash = canonical_hash([i.to_json_dict() for i in items])
        if items_hash is not None and items_hash != actual_items_hash:
            raise ValueError('stale_structured_list')
        if source_hash is not None:
            if source_hash != current_hash:
                raise ValueError('stale_structured_list')
            if items_hash == actual_items_hash:
                return items
        # Records without a payload fingerprint are checked against the answer text instead.
        text_items = parse_entity_list(prediction.answer)
        if [i.text for i in text_items] != [i.text for i in items]:
            raise ValueError('stale_structured_list')
        if any(shown.citation_ids and shown.citation_ids != item.citation_ids for shown, item in zip(text_items, items)):
            raise ValueError('stale_structured_list')
        return items
    return parse_entity_list(prediction.answer, prediction.citations)


def build_answer_bundle(answer, task_track='short_answer', task_type=None):
    """Build the shared payload without promoting malformed generation to a pipeline exception."""
    metadata = {'answer_contract': 'task-answer-v1'}
    items = None
    abstained = is_explicit_abstention(answer)
    if task_type == 'list_answer_citation' and abstained:
        metadata.update(list_format_status='computed', list_representation='abstention_marker',
                        format_protocol=LIST_FORMAT_PROTOCOL)
    if task_type == 'list_answer_citation' and not abstained:
        try:
            items = parse_entity_list(answer)
            metadata.update(list_payload_metadata(answer, items))
        except ValueError as exc:
            metadata.update(list_format_status='unavailable', list_format_reason=str(exc))
    return AnswerBundleRecord(short_answer=answer if task_track == 'short_answer' else None,
                              long_answer=answer if task_track != 'short_answer' else None,
                              abstained=abstained, list_items=items, metadata=metadata)
