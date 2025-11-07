"""
Orchestrator Service Test Suite
================================
Tests intent classification, entity extraction, and routing logic.
"""

import requests
import json
import time
from typing import Dict, Any
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

# Test configuration
ORCHESTRATOR_URL = "http://localhost:5006"
GATEWAY_URL = "http://localhost:3001"

# Test statistics
stats = {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "total_latency": 0
}


def print_header(title: str):
    """Print a formatted test section header."""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}{title.center(70)}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")


def print_test(name: str, passed: bool, details: str = ""):
    """Print test result with color coding."""
    stats["total"] += 1
    if passed:
        stats["passed"] += 1
        print(f"{Fore.GREEN}✅ PASS{Style.RESET_ALL}: {name}")
    else:
        stats["failed"] += 1
        print(f"{Fore.RED}❌ FAIL{Style.RESET_ALL}: {name}")
    
    if details:
        print(f"   {Fore.YELLOW}{details}{Style.RESET_ALL}")


def test_health_check():
    """Test health endpoint."""
    print_header("Health Check")
    
    try:
        response = requests.get(f"{ORCHESTRATOR_URL}/health", timeout=5)
        passed = response.status_code == 200 and response.json().get("status") == "healthy"
        print_test(
            "Health endpoint",
            passed,
            f"Status: {response.status_code}, Response: {response.json()}"
        )
    except Exception as e:
        print_test("Health endpoint", False, f"Error: {str(e)}")


def test_symptom_intent():
    """Test symptom intent classification."""
    print_header("Symptom Intent Classification")
    
    test_cases = [
        {
            "transcript": "عندي صداع منذ يومين",
            "expected_intent": "symptom",
            "expected_routing": "rag",
            "expected_entities": {"symptoms": ["صداع"]},
        },
        {
            "transcript": "أشعر بألم في الصدر",
            "expected_intent": "symptom",
            "expected_routing": "rag",
            "expected_entities": {"symptoms": ["ألم"], "body_parts": ["صدر"]},
        },
        {
            "transcript": "I have a headache for 2 days",
            "expected_intent": "symptom",
            "expected_routing": "rag",
            "expected_entities": {"symptoms": ["headache"]},
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        try:
            start_time = time.time()
            response = requests.post(
                f"{ORCHESTRATOR_URL}/orchestrate",
                json={"transcript": case["transcript"], "sessionId": f"test-symptom-{i}"},
                timeout=5
            )
            latency = (time.time() - start_time) * 1000
            stats["total_latency"] += latency
            
            data = response.json()
            
            # Check intent
            intent_match = data["intent"] == case["expected_intent"]
            routing_match = data["routing"] == case["expected_routing"]
            
            # Check entities (at least one expected entity should be present)
            entity_match = False
            for entity_type, expected_values in case["expected_entities"].items():
                if entity_type in data["entities"]:
                    actual_values = data["entities"][entity_type]
                    if any(exp in actual_values for exp in expected_values):
                        entity_match = True
                        break
            
            passed = intent_match and routing_match and entity_match
            
            print_test(
                f"Case {i}: '{case['transcript'][:30]}...'",
                passed,
                f"Intent: {data['intent']} (conf: {data['confidence']:.2f}), "
                f"Routing: {data['routing']}, Entities: {len(data['entities'])}, "
                f"Latency: {latency:.1f}ms"
            )
            
        except Exception as e:
            print_test(f"Case {i}", False, f"Error: {str(e)}")


def test_emergency_intent():
    """Test emergency intent classification."""
    print_header("Emergency Intent Classification")
    
    test_cases = [
        {
            "transcript": "عندي نوبة قلبية",
            "expected_intent": "emergency",
            "expected_routing": "escalate",
            "min_confidence": 0.90,
        },
        {
            "transcript": "أعاني من صعوبة في التنفس",
            "expected_intent": "emergency",
            "expected_routing": "escalate",
            "min_confidence": 0.90,
        },
        {
            "transcript": "I'm having a heart attack",
            "expected_intent": "emergency",
            "expected_routing": "escalate",
            "min_confidence": 0.90,
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        try:
            start_time = time.time()
            response = requests.post(
                f"{ORCHESTRATOR_URL}/orchestrate",
                json={"transcript": case["transcript"], "sessionId": f"test-emergency-{i}"},
                timeout=5
            )
            latency = (time.time() - start_time) * 1000
            stats["total_latency"] += latency
            
            data = response.json()
            
            intent_match = data["intent"] == case["expected_intent"]
            routing_match = data["routing"] == case["expected_routing"]
            confidence_match = data["confidence"] >= case["min_confidence"]
            
            passed = intent_match and routing_match and confidence_match
            
            print_test(
                f"Case {i}: '{case['transcript'][:30]}...'",
                passed,
                f"Intent: {data['intent']} (conf: {data['confidence']:.2f}), "
                f"Routing: {data['routing']}, Latency: {latency:.1f}ms"
            )
            
        except Exception as e:
            print_test(f"Case {i}", False, f"Error: {str(e)}")


def test_appointment_intent():
    """Test appointment intent classification."""
    print_header("Appointment Intent Classification")
    
    test_cases = [
        {
            "transcript": "أريد حجز موعد غدا",
            "expected_intent": "appointment",
            "expected_routing": "appointment_system",
            "expected_entities": {"dates": ["غدا"]},
        },
        {
            "transcript": "ممكن أحجز موعد يوم الأحد",
            "expected_intent": "appointment",
            "expected_routing": "appointment_system",
            "expected_entities": {"dates": ["الأحد"]},
        },
        {
            "transcript": "I want to book an appointment",
            "expected_intent": "appointment",
            "expected_routing": "appointment_system",
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        try:
            start_time = time.time()
            response = requests.post(
                f"{ORCHESTRATOR_URL}/orchestrate",
                json={"transcript": case["transcript"], "sessionId": f"test-appointment-{i}"},
                timeout=5
            )
            latency = (time.time() - start_time) * 1000
            stats["total_latency"] += latency
            
            data = response.json()
            
            intent_match = data["intent"] == case["expected_intent"]
            routing_match = data["routing"] == case["expected_routing"]
            
            passed = intent_match and routing_match
            
            print_test(
                f"Case {i}: '{case['transcript'][:30]}...'",
                passed,
                f"Intent: {data['intent']} (conf: {data['confidence']:.2f}), "
                f"Routing: {data['routing']}, Entities: {data['entities']}, "
                f"Latency: {latency:.1f}ms"
            )
            
        except Exception as e:
            print_test(f"Case {i}", False, f"Error: {str(e)}")


def test_prescription_intent():
    """Test prescription intent classification."""
    print_header("Prescription Intent Classification")
    
    test_cases = [
        {
            "transcript": "أحتاج وصفة دواء باراسيتامول",
            "expected_intent": "prescription",
            "expected_routing": "pharmacy",
            "expected_entities": {"medications": ["باراسيتامول"]},
        },
        {
            "transcript": "عايز أشتري دواء للصداع",
            "expected_intent": "prescription",
            "expected_routing": "pharmacy",
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        try:
            start_time = time.time()
            response = requests.post(
                f"{ORCHESTRATOR_URL}/orchestrate",
                json={"transcript": case["transcript"], "sessionId": f"test-prescription-{i}"},
                timeout=5
            )
            latency = (time.time() - start_time) * 1000
            stats["total_latency"] += latency
            
            data = response.json()
            
            intent_match = data["intent"] == case["expected_intent"]
            routing_match = data["routing"] == case["expected_routing"]
            
            passed = intent_match and routing_match
            
            print_test(
                f"Case {i}: '{case['transcript'][:30]}...'",
                passed,
                f"Intent: {data['intent']} (conf: {data['confidence']:.2f}), "
                f"Routing: {data['routing']}, Entities: {data['entities']}, "
                f"Latency: {latency:.1f}ms"
            )
            
        except Exception as e:
            print_test(f"Case {i}", False, f"Error: {str(e)}")


def test_medical_history_intent():
    """Test medical history intent classification."""
    print_header("Medical History Intent Classification")
    
    test_cases = [
        {
            "transcript": "عندي حساسية من البنسلين",
            "expected_intent": "medical_history",
            "expected_routing": "rag",
            "expected_entities": {"medications": ["البنسلين"]},
        },
        {
            "transcript": "عملت عملية جراحية السنة الماضية",
            "expected_intent": "medical_history",
            "expected_routing": "rag",
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        try:
            start_time = time.time()
            response = requests.post(
                f"{ORCHESTRATOR_URL}/orchestrate",
                json={"transcript": case["transcript"], "sessionId": f"test-history-{i}"},
                timeout=5
            )
            latency = (time.time() - start_time) * 1000
            stats["total_latency"] += latency
            
            data = response.json()
            
            intent_match = data["intent"] == case["expected_intent"]
            routing_match = data["routing"] == case["expected_routing"]
            
            passed = intent_match and routing_match
            
            print_test(
                f"Case {i}: '{case['transcript'][:30]}...'",
                passed,
                f"Intent: {data['intent']} (conf: {data['confidence']:.2f}), "
                f"Routing: {data['routing']}, Entities: {data['entities']}, "
                f"Latency: {latency:.1f}ms"
            )
            
        except Exception as e:
            print_test(f"Case {i}", False, f"Error: {str(e)}")


def test_general_intent():
    """Test general/fallback intent classification."""
    print_header("General Intent Classification")
    
    test_cases = [
        {
            "transcript": "مرحبا كيف حالك",
            "expected_intent": "general",
            "expected_routing": "direct",
        },
        {
            "transcript": "ما هو مرض السكري",
            "expected_intent": "general",
            "expected_routing": "direct",
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        try:
            start_time = time.time()
            response = requests.post(
                f"{ORCHESTRATOR_URL}/orchestrate",
                json={"transcript": case["transcript"], "sessionId": f"test-general-{i}"},
                timeout=5
            )
            latency = (time.time() - start_time) * 1000
            stats["total_latency"] += latency
            
            data = response.json()
            
            intent_match = data["intent"] == case["expected_intent"]
            routing_match = data["routing"] == case["expected_routing"]
            
            passed = intent_match and routing_match
            
            print_test(
                f"Case {i}: '{case['transcript'][:30]}...'",
                passed,
                f"Intent: {data['intent']} (conf: {data['confidence']:.2f}), "
                f"Routing: {data['routing']}, Latency: {latency:.1f}ms"
            )
            
        except Exception as e:
            print_test(f"Case {i}", False, f"Error: {str(e)}")


def test_entity_extraction():
    """Test entity extraction accuracy."""
    print_header("Entity Extraction")
    
    test_cases = [
        {
            "transcript": "عندي صداع منذ 3 أيام في الرأس",
            "expected_entities": {
                "symptoms": ["صداع"],
                "body_parts": ["رأس"],
                "durations": ["3 أيام"]
            },
        },
        {
            "transcript": "أحتاج باراسيتامول للألم في الظهر",
            "expected_entities": {
                "medications": ["باراسيتامول"],
                "body_parts": ["ظهر"],
                "symptoms": ["ألم"]
            },
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        try:
            response = requests.post(
                f"{ORCHESTRATOR_URL}/orchestrate",
                json={"transcript": case["transcript"], "sessionId": f"test-entity-{i}"},
                timeout=5
            )
            
            data = response.json()
            entities = data["entities"]
            
            # Check each expected entity type
            matches = 0
            total_expected = sum(len(v) for v in case["expected_entities"].values())
            
            for entity_type, expected_values in case["expected_entities"].items():
                if entity_type in entities:
                    actual_values = entities[entity_type]
                    for exp in expected_values:
                        if exp in actual_values:
                            matches += 1
            
            passed = matches >= (total_expected * 0.5)  # At least 50% match
            
            print_test(
                f"Case {i}: '{case['transcript'][:40]}...'",
                passed,
                f"Extracted: {entities}, Matches: {matches}/{total_expected}"
            )
            
        except Exception as e:
            print_test(f"Case {i}", False, f"Error: {str(e)}")


def test_latency_performance():
    """Test orchestrator latency."""
    print_header("Latency Performance")
    
    latencies = []
    
    for i in range(10):
        try:
            start_time = time.time()
            response = requests.post(
                f"{ORCHESTRATOR_URL}/orchestrate",
                json={"transcript": "عندي صداع", "sessionId": f"test-latency-{i}"},
                timeout=5
            )
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            
        except Exception as e:
            print(f"{Fore.RED}Request {i+1} failed: {str(e)}{Style.RESET_ALL}")
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        # Target: <50ms average
        passed = avg_latency < 50
        
        print_test(
            "Average latency",
            passed,
            f"Avg: {avg_latency:.1f}ms, Min: {min_latency:.1f}ms, Max: {max_latency:.1f}ms (Target: <50ms)"
        )


def test_gateway_integration():
    """Test orchestrator via gateway."""
    print_header("Gateway Integration")
    
    try:
        response = requests.post(
            f"{GATEWAY_URL}/llm/orchestrate",
            json={"transcript": "عندي صداع", "sessionId": "test-gateway"},
            timeout=5
        )
        
        passed = response.status_code == 200 or response.status_code == 201
        data = response.json() if passed else {}
        
        print_test(
            "Gateway /llm/orchestrate endpoint",
            passed,
            f"Status: {response.status_code}, Response: {data}"
        )
        
    except Exception as e:
        print_test("Gateway integration", False, f"Error: {str(e)}")


def print_summary():
    """Print test summary."""
    print_header("Test Summary")
    
    pass_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
    avg_latency = (stats["total_latency"] / stats["total"]) if stats["total"] > 0 else 0
    
    print(f"{Fore.CYAN}Total Tests:{Style.RESET_ALL} {stats['total']}")
    print(f"{Fore.GREEN}Passed:{Style.RESET_ALL} {stats['passed']}")
    print(f"{Fore.RED}Failed:{Style.RESET_ALL} {stats['failed']}")
    print(f"{Fore.YELLOW}Pass Rate:{Style.RESET_ALL} {pass_rate:.1f}%")
    print(f"{Fore.YELLOW}Average Latency:{Style.RESET_ALL} {avg_latency:.1f}ms\n")
    
    if pass_rate >= 90:
        print(f"{Fore.GREEN}{'='*70}")
        print(f"{Fore.GREEN}✅ ORCHESTRATOR TESTS PASSED! All systems operational.{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*70}\n")
    else:
        print(f"{Fore.RED}{'='*70}")
        print(f"{Fore.RED}❌ SOME TESTS FAILED. Please review the errors above.{Style.RESET_ALL}")
        print(f"{Fore.RED}{'='*70}\n")


if __name__ == "__main__":
    print(f"\n{Fore.MAGENTA}{'='*70}")
    print(f"{Fore.MAGENTA}{'ORCHESTRATOR SERVICE TEST SUITE'.center(70)}")
    print(f"{Fore.MAGENTA}{'='*70}{Style.RESET_ALL}\n")
    
    print(f"{Fore.YELLOW}Testing orchestrator at: {ORCHESTRATOR_URL}")
    print(f"{Fore.YELLOW}Testing gateway at: {GATEWAY_URL}{Style.RESET_ALL}\n")
    
    # Run all test suites
    test_health_check()
    test_symptom_intent()
    test_emergency_intent()
    test_appointment_intent()
    test_prescription_intent()
    test_medical_history_intent()
    test_general_intent()
    test_entity_extraction()
    test_latency_performance()
    test_gateway_integration()
    
    # Print summary
    print_summary()
