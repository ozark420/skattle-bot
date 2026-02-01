import os
import requests
import time

API_BASE = 'https://api.bankr.bot'
API_KEY = os.environ.get('BANKR_API_KEY', 'bk_P623F43XRVFE4Z6BLZ2M4Y86DBYNHCU6')
HEADERS = {'Content-Type': 'application/json', 'X-API-Key': API_KEY}

# Check if skattlebot.base.eth is available
res = requests.post(f'{API_BASE}/agent/prompt', headers=HEADERS, json={
    'prompt': 'Check if skattlebot.base.eth basename is available for registration on Base. If available, tell me the price to register it for 1 year.'
})
job_id = res.json().get('jobId')
print(f'Job: {job_id}')

for _ in range(30):
    time.sleep(2)
    r = requests.get(f'{API_BASE}/agent/job/{job_id}', headers=HEADERS)
    data = r.json()
    status = data.get('status')
    print(f'Status: {status}')
    if status == 'completed':
        print(f"Response: {data.get('response', '')}")
        break
    elif status == 'failed':
        print('Failed')
        break
