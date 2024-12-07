# Authentication Guide

## Overview
The Energy Forecast API uses a dual authentication system:
1. API Keys for service authentication
2. JWT tokens for user authentication

## API Key Authentication

### Obtaining an API Key
1. Register at the developer portal
2. Create a new application
3. Generate API key

### Using API Keys
```python
import requests

headers = {
    "X-API-Key": "your-api-key"
}

response = requests.get(
    "http://localhost:8000/forecast/cities",
    headers=headers
)
```

### API Key Best Practices
- Keep keys secure
- Rotate regularly
- Use environment variables
- Different keys per environment

## JWT Authentication

### Getting a Token
```python
import requests

response = requests.post(
    "http://localhost:8000/auth/token",
    json={
        "username": "user@example.com",
        "password": "secure_password"
    }
)

token = response.json()["access_token"]
```

### Using JWT Tokens
```python
headers = {
    "Authorization": f"Bearer {token}"
}

response = requests.post(
    "http://localhost:8000/forecast/demand",
    headers=headers,
    json=forecast_request
)
```

### Token Management
- Tokens expire after 30 minutes
- Refresh tokens available
- Automatic token refresh
- Token blacklisting

## Rate Limiting

### Limits
- 100 requests per minute per API key
- 1000 requests per hour per user
- Burst allowance: 10 requests

### Headers
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1635789600
```

### Handling Rate Limits
```python
def make_request_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        
        if response.status_code == 429:  # Rate limit exceeded
            wait_time = int(response.headers.get('Retry-After', 60))
            time.sleep(wait_time)
            continue
            
        return response
        
    raise Exception("Max retries exceeded")
```

## Security Best Practices

### API Key Security
1. Store securely
   ```python
   from decouple import config
   
   api_key = config('API_KEY')
   ```

2. Different keys per environment
   ```python
   if environment == 'production':
       api_key = config('PROD_API_KEY')
   else:
       api_key = config('DEV_API_KEY')
   ```

3. Regular rotation
   ```python
   # Check key age
   key_age = (datetime.now() - key_creation_date).days
   if key_age > 90:  # Rotate every 90 days
       new_key = rotate_api_key()
   ```

### JWT Security
1. Secure token storage
   ```javascript
   // Browser
   localStorage.setItem('token', jwt_token)
   ```

2. Token refresh
   ```python
   def refresh_token(refresh_token):
       response = requests.post(
           "http://localhost:8000/auth/refresh",
           json={"refresh_token": refresh_token}
       )
       return response.json()["access_token"]
   ```

3. Token validation
   ```python
   def validate_token(token):
       try:
           payload = jwt.decode(
               token,
               settings.SECRET_KEY,
               algorithms=["HS256"]
           )
           return payload
       except jwt.ExpiredSignatureError:
           return None
   ```

## Error Handling

### Authentication Errors
```python
def handle_auth_error(response):
    if response.status_code == 401:
        if "token_expired" in response.text:
            new_token = refresh_token()
            # Retry with new token
        elif "invalid_token" in response.text:
            # Redirect to login
        elif "invalid_api_key" in response.text:
            # Log error and alert
```

### Rate Limit Errors
```python
def handle_rate_limit(response):
    if response.status_code == 429:
        wait_time = int(response.headers['Retry-After'])
        time.sleep(wait_time)
        # Retry request
```

## Testing

### API Key Tests
```python
def test_invalid_api_key():
    headers = {"X-API-Key": "invalid_key"}
    response = client.get("/forecast/cities", headers=headers)
    assert response.status_code == 401

def test_rate_limit():
    for _ in range(101):  # Exceed limit
        client.get("/forecast/cities", headers=headers)
    assert response.status_code == 429
```

### JWT Tests
```python
def test_token_expiry():
    # Create expired token
    expired_token = create_expired_token()
    headers = {"Authorization": f"Bearer {expired_token}"}
    response = client.get("/protected", headers=headers)
    assert response.status_code == 401
```

## Monitoring

### Authentication Metrics
- Failed attempts
- Token refresh rate
- API key usage
- Rate limit hits

### Alerts
- Multiple failed attempts
- Unusual API key usage
- Rate limit threshold
- Token refresh spikes

## Support

### Common Issues
1. Invalid API Key
   - Check key format
   - Verify environment
   - Check expiration

2. Token Expired
   - Implement auto-refresh
   - Check token lifetime
   - Verify clock sync

3. Rate Limit
   - Implement backoff
   - Check usage patterns
   - Consider upgrading

### Contact
- Technical support: support@energyforecast.com
- Security issues: security@energyforecast.com
- Documentation: docs@energyforecast.com
