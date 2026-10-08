"""Upload a private project reference to DashScope's temporary OSS storage.

Contract: https://www.alibabacloud.com/help/en/model-studio/get-temporary-file-url
The worker validates the returned public upload host and strips model API auth
before uploading. The OSS resource is private and has a temporary lifetime.
"""

import os
from pathlib import Path


def temporary_file(path: str, model: str) -> str:
    import requests

    host = "dashscope-intl.aliyuncs.com" if os.environ.get("DASHSCOPE_REGION") == "intl" else "dashscope.aliyuncs.com"
    response = requests.get(f"https://{host}/api/v1/uploads", params={"action": "getPolicy", "model": model}, headers={"Authorization": f"Bearer {os.environ['DASHSCOPE_API_KEY']}"}, timeout=30)
    response.raise_for_status()
    policy = response.json()["data"]
    filename = Path(path).name
    key = policy["upload_dir"] + "/" + filename
    with Path(path).open("rb") as source:
        fields = {"OSSAccessKeyId": (None, policy["oss_access_key_id"]), "Signature": (None, policy["signature"]), "policy": (None, policy["policy"]), "key": (None, key), "success_action_status": (None, "200"), "x-oss-object-acl": (None, policy["x_oss_object_acl"]), "x-oss-forbid-overwrite": (None, policy["x_oss_forbid_overwrite"]), "file": (filename, source)}
        uploaded = requests.post(policy["upload_host"], files=fields, timeout=120)
    uploaded.raise_for_status()
    return f"oss://{key}"
