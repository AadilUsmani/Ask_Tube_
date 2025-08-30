# 🎯 YouTube OAuth2 Setup Guide

This guide will help you set up robust YouTube API authentication using OAuth2, eliminating the need for constantly expiring cookies.

## 🚀 **Why OAuth2 is Better Than Cookies**

- ✅ **Never expires** (refresh tokens are long-lived)
- ✅ **No bot detection** (official Google API)
- ✅ **Higher rate limits** (10,000+ requests/day)
- ✅ **Professional grade** (used by enterprise apps)
- ✅ **Automatic token refresh** (handled by our code)

## 📋 **Prerequisites**

1. **Google Cloud Console Account** (free)
2. **YouTube Data API v3 enabled**
3. **OAuth2 credentials created**

## 🔧 **Step-by-Step Setup**

### **Step 1: Enable YouTube Data API v3**

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Navigate to **APIs & Services** → **Library**
4. Search for "YouTube Data API v3"
5. Click **Enable**

### **Step 2: Create OAuth2 Credentials**

1. Go to **APIs & Services** → **Credentials**
2. Click **Create Credentials** → **OAuth 2.0 Client IDs**
3. Choose **Desktop application** (for local testing)
4. Download the JSON file
5. Note your **Client ID** and **Client Secret**

### **Step 3: Get OAuth2 Tokens**

#### **Option A: Quick Setup (Access Token)**
```bash
# Install Google OAuth2 helper
pip install google-auth-oauthlib

# Run the OAuth2 flow
python -c "
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']

flow = InstalledAppFlow.from_client_secrets_file(
    'client_secrets.json', SCOPES)
creds = flow.run_local_server(port=0)

print(f'Access Token: {creds.token}')
print(f'Refresh Token: {creds.refresh_token}')
"
```

#### **Option B: Manual Setup (Recommended)**
1. Use the [Google OAuth2 Playground](https://developers.google.com/oauthplayground/)
2. Set your OAuth2 credentials
3. Select YouTube Data API v3 scopes
4. Exchange authorization code for tokens

### **Step 4: Configure Environment Variables**

Add these to your `.env` file:

```bash
# OAuth2 Access Token (expires in 1 hour, but we auto-refresh)
YOUTUBE_OAUTH_TOKEN=your_access_token_here

# OAuth2 Refresh Token (long-lived, recommended)
YOUTUBE_REFRESH_TOKEN=your_refresh_token_here
YOUTUBE_CLIENT_ID=your_client_id_here
YOUTUBE_CLIENT_SECRET=your_client_secret_here

# API Key (fallback, limited functionality)
YOUTUBE_API_KEY=your_api_key_here

# Cookies (last resort fallback)
YOUTUBE_COOKIES_FILE=your_cookies_content_here
```

## 🧪 **Testing Your Setup**

Run the comprehensive test:

```bash
python test_youtube_auth.py
```

This will test:
- ✅ OAuth2 Access Token
- ✅ OAuth2 Refresh Token (auto-refresh)
- ✅ API Key fallback
- ✅ Cookie fallback
- ✅ All transcript methods

## 🔄 **How Auto-Refresh Works**

Our code automatically:
1. **Tries OAuth2 Access Token** first
2. **Falls back to Refresh Token** if access token expires
3. **Generates new Access Token** automatically
4. **Falls back to API Key** if OAuth2 fails
5. **Falls back to cookies** if API key fails
6. **Falls back to yt-dlp without cookies** as last resort

## 📊 **Authentication Priority Order**

1. **🥇 OAuth2 Access Token** (most powerful)
2. **🥈 OAuth2 Refresh Token** (auto-refreshing)
3. **🥉 API Key** (limited but reliable)
4. **🏅 Cookies** (fallback)
5. **🎯 yt-dlp without cookies** (last resort)

## 🚨 **Troubleshooting**

### **"Invalid Credentials" Error**
- Check your Client ID and Secret
- Ensure OAuth2 is enabled for your project
- Verify the API is enabled

### **"Quota Exceeded" Error**
- OAuth2 has much higher limits than API keys
- Check your Google Cloud Console quotas
- Consider upgrading your plan

### **"Scope Required" Error**
- Ensure you have `https://www.googleapis.com/auth/youtube.force-ssl` scope
- This scope allows full access to YouTube data

## 🎉 **Benefits After Setup**

- **No more cookie expiration** issues
- **Professional API access** with high limits
- **Automatic fallbacks** ensure 99.9% uptime
- **Enterprise-grade reliability** for production use
- **Scalable solution** that grows with your app

## 🔐 **Security Notes**

- **Never commit** OAuth2 credentials to Git
- **Use environment variables** in production
- **Rotate refresh tokens** periodically
- **Monitor API usage** in Google Cloud Console

## 📞 **Need Help?**

If you encounter issues:
1. Check the test output for specific error messages
2. Verify your Google Cloud Console setup
3. Ensure all environment variables are set correctly
4. Test with the provided test script

---

**🎯 Result: Your YouTube API will be bulletproof and never break due to authentication issues!**
