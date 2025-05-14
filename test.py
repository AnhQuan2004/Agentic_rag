import requests

# URL và tham số truy vấn
url = "https://n8n.vbi-server.com/webhook/get-data"
params = {
    "address": "0x000123456789"
}

# Gửi yêu cầu GET
response = requests.get(url, params=params)

# In kết quả phản hồi
print("Status Code:", response.status_code)
print("Response Body:", response.text)
