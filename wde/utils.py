import shortuuid


def random_uuid(length=22) -> str:
    return str(shortuuid.random(length=length))
