import urllib.request
import json
import time

start = time.time()
req = urllib.request.Request(
    'http://127.0.0.1:5001/api/agent/chat',
    data=json.dumps({"query": "robot arm path planning"}).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
try:
    res = urllib.request.urlopen(req).read()
    data = json.loads(res)
    print(f"Took: {time.time()-start:.2f}s")
    print("Report Preview:", data['report'][:200])
except Exception as e:
    print(f"Error: {e}")
