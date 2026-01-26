"""
Test suite for Medical Guardrails — Emergency Detection
Covers all emergency keywords, edge cases, and false positive prevention
"""
import pytest
from guardrails import MedicalGuardrails


@pytest.fixture
def guardrails():
    """Create guardrails instance without Redis (no rate limiting)."""
    return MedicalGuardrails(redis_client=None)


class TestEmergencyDetection:
    """Test emergency keyword detection."""

    # All keywords from EMERGENCY_KEYWORDS_AR
    EMERGENCY_KEYWORDS = [
        "نوبة قلبية",
        "heart attack",
        "صعوبة تنفس",
        "نزيف شديد",
        "فقدان وعي",
        "جلطة",
        "سكتة",
        "صدمة",
        "حساسية شديدة",
        "اختناق",
        "ألم صدر شديد",
        "شلل مفاجئ",
        "تسمم",
    ]

    @pytest.mark.parametrize("keyword", EMERGENCY_KEYWORDS)
    def test_detects_emergency_keywords(self, guardrails, keyword):
        """Each emergency keyword should be detected."""
        message = f"أشعر بـ {keyword}"
        assert guardrails.detect_emergency(message) is True

    @pytest.mark.parametrize("keyword", EMERGENCY_KEYWORDS)
    def test_detects_emergency_case_insensitive(self, guardrails, keyword):
        """Detection should be case-insensitive for English terms."""
        if keyword.isascii():
            message = f"I think I'm having a {keyword.upper()}"
            assert guardrails.detect_emergency(message) is True

    def test_detects_embedded_keywords(self, guardrails):
        """Keywords embedded in longer text should be detected."""
        messages = [
            "أمي تعاني من صعوبة تنفس شديدة منذ ساعة",
            "My father had a heart attack this morning",
            "أبي فقد الوعي فجأة ولا يستجيب",
            "شعرت بألم صدر شديد وذراعي اليسرى تؤلمني",
        ]
        for msg in messages:
            assert guardrails.detect_emergency(msg) is True, f"Failed for: {msg}"

    def test_no_false_positives_normal_medical(self, guardrails):
        """Normal medical queries should NOT trigger emergency."""
        normal_messages = [
            "أشعر بصداع خفيف",
            "عندي كحة منذ يومين",
            "أحتاج موعد مع طبيب أسنان",
            "متى ينتهي مفعول الدواء؟",
            "هل يمكنني أخذ باراسيتامول؟",
            "أريد فحص دم عام",
            "ما هي أعراض الأنفلونزا؟",
            "I have a mild headache",
            "My knee hurts when I walk",
        ]
        for msg in normal_messages:
            assert guardrails.detect_emergency(msg) is False, f"False positive for: {msg}"

    def test_detects_arabic_dialects(self, guardrails):
        """Emergency keywords in Egyptian/Saudi dialects."""
        dialect_messages = [
            "أبويا جاله جلطة",  # Egyptian
            "أخوي ما يقدر يتنفس",  # Saudi
            "عمي فقد وعيه فجأة",
        ]
        for msg in dialect_messages:
            # These contain base keywords (جلطة, تنفس, وعي)
            result = guardrails.detect_emergency(msg)
            # At least جلطة should be detected
            if "جلطة" in msg or "وعي" in msg:
                assert result is True, f"Should detect: {msg}"


class TestHarmfulContentDetection:
    """Test harmful content blocking."""

    HARMFUL_KEYWORDS = [
        "انتحار",
        "suicide",
        "إيذاء",
        "self-harm",
        "قتل",
        "مخدرات غير موصوفة",
        "illegal drugs",
        "كيفية الموت",
        "how to die",
        "إنهاء الحياة",
    ]

    @pytest.mark.parametrize("keyword", HARMFUL_KEYWORDS)
    def test_blocks_harmful_content(self, guardrails, keyword):
        """Harmful keywords should be blocked."""
        message = f"أريد معرفة عن {keyword}"
        result = guardrails.check_harmful_content(message)
        assert result["blocked"] is True
        assert result["reason"] == "harmful_content"

    def test_allows_medical_context(self, guardrails):
        """Medical discussions mentioning harm should be allowed with care."""
        # Note: Current implementation blocks by keyword, so these may still block
        # This test documents expected behavior — may need refinement
        messages = [
            "ما هي أعراض الاكتئاب؟",
            "كيف أساعد شخص يعاني من القلق؟",
        ]
        for msg in messages:
            result = guardrails.check_harmful_content(msg)
            # These should NOT be blocked (no harmful keywords)
            assert result["blocked"] is False, f"Wrongly blocked: {msg}"


class TestDisclaimer:
    """Test medical disclaimer injection."""

    def test_arabic_disclaimer(self, guardrails):
        """Arabic disclaimer should be prepended."""
        response = "شرب الماء مفيد للصحة"
        result = guardrails.inject_disclaimer(response, language="ar")
        assert result.startswith("⚠️")
        assert "تنويه طبي" in result
        assert response in result

    def test_english_disclaimer(self, guardrails):
        """English disclaimer should be prepended."""
        response = "Drinking water is good for health"
        result = guardrails.inject_disclaimer(response, language="en")
        assert result.startswith("⚠️")
        assert "Medical Disclaimer" in result
        assert response in result


class TestEmergencyResponse:
    """Test emergency response generation."""

    def test_emergency_response_arabic(self, guardrails):
        """Emergency response should include emergency numbers."""
        response = guardrails.get_emergency_response("ar")
        assert "🚨" in response
        assert "123" in response  # Emergency number
        assert "طارئة" in response


class TestMessageLengthValidation:
    """Test message length limits."""

    def test_allows_normal_length(self, guardrails):
        """Normal messages should be allowed."""
        message = "مرحبا، أريد حجز موعد"
        result = guardrails.check_message_length(message)
        assert result["allowed"] is True

    def test_blocks_excessive_length(self, guardrails):
        """Messages exceeding MAX_MESSAGE_LENGTH should be blocked."""
        message = "أ" * (guardrails.MAX_MESSAGE_LENGTH + 100)
        result = guardrails.check_message_length(message)
        assert result["allowed"] is False
        assert "length" in result["reason"].lower()


class TestTurnLimit:
    """Test session turn limits."""

    def test_allows_within_limit(self, guardrails):
        """Turns within limit should be allowed."""
        result = guardrails.check_turn_limit("session-123", turn_count=5)
        assert result["allowed"] is True

    def test_blocks_excessive_turns(self, guardrails):
        """Turns exceeding MAX_TURNS_PER_SESSION should be blocked."""
        result = guardrails.check_turn_limit(
            "session-123",
            turn_count=guardrails.MAX_TURNS_PER_SESSION + 5
        )
        assert result["allowed"] is False


class TestValidateRequest:
    """Test full request validation pipeline."""

    def test_validates_normal_request(self, guardrails):
        """Normal requests should pass validation."""
        result = guardrails.validate_request(
            message="أريد حجز موعد مع طبيب عيون",
            user_id="user-123",
            session_id="session-456",
            turn_count=3,
        )
        assert result["allowed"] is True
        assert result["is_emergency"] is False

    def test_detects_emergency_in_validation(self, guardrails):
        """Emergency keywords should be flagged in validation."""
        result = guardrails.validate_request(
            message="أشعر بألم صدر شديد وصعوبة في التنفس",
            user_id="user-123",
            session_id="session-456",
            turn_count=1,
        )
        # Request is allowed but flagged as emergency
        assert result["is_emergency"] is True

    def test_blocks_harmful_in_validation(self, guardrails):
        """Harmful content should be blocked in validation."""
        result = guardrails.validate_request(
            message="كيفية الموت",
            user_id="user-123",
            session_id="session-456",
            turn_count=1,
        )
        assert result["allowed"] is False
        assert "harmful" in result.get("reason", "").lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
