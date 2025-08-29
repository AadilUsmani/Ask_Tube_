#!/usr/bin/env python3
"""
YouTube Cookies Upload Script for Render
This script helps you upload YouTube cookies to Render environment variables.
"""

import os
import requests
import json
from pathlib import Path

def read_cookies_file(file_path: str) -> str:
    """Read cookies file and return content as string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ Cookies file not found: {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error reading cookies file: {e}")
        return None

def upload_to_render(cookies_content: str, render_token: str, service_id: str) -> bool:
    """Upload cookies to Render environment variables."""
    
    # Render API endpoint
    url = f"https://api.render.com/v1/services/{service_id}/env-vars"
    
    headers = {
        "Authorization": f"Bearer {render_token}",
        "Content-Type": "application/json"
    }
    
    # Create environment variable for cookies
    data = {
        "key": "YOUTUBE_COOKIES_FILE",
        "value": cookies_content,
        "sync": False  # Don't sync this sensitive data
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 201:
            print("✅ Successfully uploaded cookies to Render!")
            return True
        else:
            print(f"❌ Failed to upload cookies: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error uploading to Render: {e}")
        return False

def main():
    print("🍪 YouTube Cookies Upload Script for Render")
    print("=" * 50)
    
    # Check if cookies file exists
    cookies_file = "youtube_cookies.txt"
    if not Path(cookies_file).exists():
        print(f"❌ Cookies file '{cookies_file}' not found!")
        print("Please:")
        print("1. Export cookies from your browser")
        print("2. Save them as 'youtube_cookies.txt'")
        print("3. Run this script again")
        return
    
    # Read cookies
    cookies_content = read_cookies_file(cookies_file)
    if not cookies_content:
        return
    
    print(f"✅ Found cookies file: {cookies_file}")
    print(f"📏 Cookies content length: {len(cookies_content)} characters")
    
    # Get Render credentials
    print("\n🔑 Render Configuration:")
    render_token = input("Enter your Render API token: ").strip()
    service_id = input("Enter your Render service ID: ").strip()
    
    if not render_token or not service_id:
        print("❌ Render token and service ID are required!")
        return
    
    # Upload to Render
    print("\n🚀 Uploading cookies to Render...")
    success = upload_to_render(cookies_content, render_token, service_id)
    
    if success:
        print("\n🎉 Setup complete!")
        print("Your Render service should now be able to access YouTube with authentication.")
        print("\nNext steps:")
        print("1. Restart your Render service")
        print("2. Test the API endpoints again")
        print("3. The bot detection should be bypassed!")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
