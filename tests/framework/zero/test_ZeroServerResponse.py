from wde.microservices.framework.zero.schema import ZeroMSQ


def test_ZeroServerResponseOk():
    from wde.microservices.framework.zero.schema import ZeroServerResponseOk
    rep = ZeroServerResponseOk()

    data, payload = ZeroMSQ.load(rep)

    assert data == b'{"msg": {"state": "ok", "msg": ""}}'
    assert payload == []


def test_ZeroServerResponseError():
    from wde.microservices.framework.zero.schema import ZeroServerResponseError
    rep = ZeroServerResponseError()

    data, payload = ZeroMSQ.load(rep)

    assert data == b'{"msg": {"state": "error", "msg": ""}}'
    assert payload == []


def test_ZeroServerStreamResponseOk():
    from wde.microservices.framework.zero.schema import \
        ZeroServerStreamResponseOk
    rep = ZeroServerStreamResponseOk(snd_more=True, rep_id=0)

    data, payload = ZeroMSQ.load(rep)

    assert data == b'{"msg": {"state": "ok", "msg": "", "snd_more": true, "rep_id": 0}}'
    assert payload == []

    rep = ZeroServerStreamResponseOk(snd_more=False, rep_id=1)

    data, payload = ZeroMSQ.load(rep)

    assert data == b'{"msg": {"state": "ok", "msg": "", "snd_more": false, "rep_id": 1}}'
    assert payload == []
