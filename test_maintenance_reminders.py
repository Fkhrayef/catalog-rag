#!/usr/bin/env python3
"""
Test script for the maintenance reminder system
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_documents_info():
    """Test documents info endpoint"""
    print("\n📚 Testing Documents Info...")
    try:
        response = requests.get(f"{BASE_URL}/documents/info")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_maintenance_reminders():
    """Test maintenance reminders endpoint"""
    print("\n🔧 Testing Maintenance Reminders...")
    
    # Test cases with different mileage values
    test_cases = [
        {
            "document_name": "Nissan Altima Manual",
            "current_mileage": 44000,  # 44,000 km
            "user_id": "test_user_1"
        },
        {
            "document_name": "Nissan Sentra Manual", 
            "current_mileage": 25000,  # 25,000 km
            "user_id": "test_user_2"
        },
        {
            "document_name": "Nissan Altima Manual",
            "current_mileage": 15000,  # 15,000 km
            "user_id": "test_user_3"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Document: {test_case['document_name']}")
        print(f"Mileage: {test_case['current_mileage']:,} km")
        
        try:
            response = requests.post(
                f"{BASE_URL}/generate-maintenance-reminders",
                headers={"Content-Type": "application/json"},
                json=test_case
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Success: {data.get('success', False)}")
                print(f"Reminders Count: {len(data.get('reminders', []))}")
                
                # Display reminders
                for j, reminder in enumerate(data.get('reminders', []), 1):
                    print(f"  {j}. {reminder.get('message', 'N/A')}")
                    print(f"     Due: {reminder.get('dueDate', 'N/A')}")
                    print(f"     Priority: {reminder.get('priority', 'N/A')}")
                    print(f"     Category: {reminder.get('category', 'N/A')}")
                    print(f"     Mileage: {reminder.get('mileage', 'N/A'):,} km")
                    print()
            else:
                print(f"❌ Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def test_error_cases():
    """Test error handling"""
    print("\n🚨 Testing Error Cases...")
    
    # Test with non-existent document
    print("\n--- Test: Non-existent Document ---")
    try:
        response = requests.post(
            f"{BASE_URL}/generate-maintenance-reminders",
            headers={"Content-Type": "application/json"},
            json={
                "document_name": "Non-existent Manual",
                "current_mileage": 30000,
                "user_id": "test_user"
            }
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test with invalid mileage
    print("\n--- Test: Invalid Mileage ---")
    try:
        response = requests.post(
            f"{BASE_URL}/generate-maintenance-reminders",
            headers={"Content-Type": "application/json"},
            json={
                "document_name": "Nissan Altima Manual",
                "current_mileage": -1000,  # Invalid negative mileage
                "user_id": "test_user"
            }
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting Maintenance Reminder System Tests")
    print("=" * 50)
    
    # Test basic endpoints
    health_ok = test_health()
    docs_ok = test_documents_info()
    
    if health_ok and docs_ok:
        print("\n✅ Basic endpoints working, proceeding with maintenance tests...")
        
        # Test maintenance reminders
        test_maintenance_reminders()
        
        # Test error cases
        test_error_cases()
        
        print("\n🎉 All tests completed!")
    else:
        print("\n❌ Basic endpoints failed. Please check your server.")

if __name__ == "__main__":
    main()
