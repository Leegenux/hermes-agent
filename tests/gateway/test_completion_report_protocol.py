"""Strict completion output contracts; never infer reports from natural language."""
import json
import pytest


def test_receipt_keeps_side_and_binds_report_ids():
    from gateway.completion_report_protocol import parse_receipt
    value = {'v': 1, 'side_response': 'answer', 'reports': [{'identity': 'a', 'text': 'verified result'}], 'unreported': ['b']}
    parsed = parse_receipt(json.dumps(value), ['a', 'b'])
    assert parsed['side_response'] == 'answer'
    assert parsed['reports'] == [{'identity': 'a', 'text': 'verified result'}]
    assert parsed['unreported'] == ['b']
    assert parsed['text'] == 'answer\n\nverified result'


@pytest.mark.parametrize('change', ['unknown', 'duplicate', 'overlap', 'missing', 'empty_report', 'version_bool', 'extra', 'media'])
def test_invalid_receipt_is_not_acknowledged(change):
    from gateway.completion_report_protocol import parse_receipt, ReceiptError
    value = {'v': 1, 'side_response': 'side', 'reports': [], 'unreported': ['a']}
    if change == 'unknown': value['unreported'] = ['alien']
    if change == 'duplicate': value['unreported'] = ['a', 'a']
    if change == 'overlap': value['reports'] = [{'identity': 'a', 'text': 'report'}]
    if change == 'missing': value['unreported'] = []
    if change == 'empty_report': value['reports'] = [{'identity': 'a', 'text': ' '}]; value['unreported'] = []
    if change == 'version_bool': value['v'] = True
    if change == 'extra': value['confirmed'] = True
    if change == 'media': value['side_response'] = 'MEDIA:C:/file'
    with pytest.raises(ReceiptError): parse_receipt(json.dumps(value), ['a'])


def test_plain_text_and_duplicate_json_keys_are_not_receipts():
    from gateway.completion_report_protocol import parse_receipt, ReceiptError
    for text in ('already reported', '{"v":1,"v":1,"side_response":"x","reports":[],"unreported":["a"]}'):
        with pytest.raises(ReceiptError): parse_receipt(text, ['a'])
