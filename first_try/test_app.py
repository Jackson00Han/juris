# test_app.py
import requests, json

test_pdf      = 'main12.pdf'       # ← make sure this file exists
test_question = 'how many chapters are there in the book?'
url           = 'http://127.0.0.1:5000/ask'

with open(test_pdf, 'rb') as f:
    resp = requests.post(
        url,
        files={'file': (test_pdf, f, 'application/pdf')},
        data={'question': test_question}
    )

print(json.dumps(resp.json(), indent=2))
