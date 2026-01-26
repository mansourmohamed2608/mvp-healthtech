"""
Medical Policy Guardrails for HealthTech AI
============================================
Implements safety checks, disclaimers, rate limiting, and content filtering
for medical conversations.
"""

from typing import Dict, Optional, List, Any
import re
import os
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MedicalGuardrails:
    """
    Enforces medical safety policies and content guidelines.
    """

    # Emergency keywords that require immediate escalation
    EMERGENCY_KEYWORDS_AR = [
        "نوبة قلبية", "heart attack", "صعوبة تنفس", "نزيف شديد",
        "فقدان وعي", "جلطة", "سكتة", "صدمة", "حساسية شديدة",
        "اختناق", "ألم صدر شديد", "شلل مفاجئ", "تسمم"
    ]

    # Harmful content patterns to block
    HARMFUL_TOPICS = [
        "انتحار", "suicide", "إيذاء", "self-harm", "قتل", "مخدرات غير موصوفة",
        "illegal drugs", "كيفية الموت", "how to die", "إنهاء الحياة"
    ]

    # Medical disclaimer templates
    DISCLAIMER_AR = (
        "⚠️ **تنويه طبي مهم**: أنا مساعد ذكي وليس طبيبًا بشريًا. "
        "المعلومات المقدمة لأغراض تعليمية فقط. يرجى استشارة طبيب مؤهل "
        "للحصول على تشخيص وعلاج دقيق.\n\n"
    )

    DISCLAIMER_EN = (
        "⚠️ **Important Medical Disclaimer**: I am an AI assistant, not a human doctor. "
        "The information provided is for educational purposes only. Please consult "
        "a qualified healthcare professional for accurate diagnosis and treatment.\n\n"
    )

    EMERGENCY_RESPONSE_AR = (
        "🚨 **حالة طارئة محتملة**: تم رصد أعراض قد تكون خطيرة. "
        "يرجى الاتصال بالإسعاف فورًا على رقم 123 أو التوجه إلى أقرب مستشفى. "
        "لا تنتظر وابحث عن مساعدة طبية فورية.\n\n"
        "📞 أرقام الطوارئ:\n"
        "- الإسعاف: 123\n"
        "- الشرطة: 122\n"
        "- الدفاع المدني: 125\n"
    )

    MAX_TURNS_PER_SESSION = 20  # Maximum conversation turns
    MAX_MESSAGE_LENGTH = 2000  # Maximum characters per message

    def __init__(self, redis_client=None):
        """
        Initialize guardrails with optional Redis for rate limiting.

        Args:
            redis_client: Redis client for distributed rate limiting (optional)
        """
        self.redis_client = redis_client
        logger.info("MedicalGuardrails initialized")

    def inject_disclaimer(self, response: str, language: str = "ar") -> str:
        """
        Adds medical disclaimer to AI responses.

        Args:
            response: The AI-generated response
            language: Language code ('ar' or 'en')

        Returns:
            Response with disclaimer prepended
        """
        disclaimer = self.DISCLAIMER_AR if language == "ar" else self.DISCLAIMER_EN
        return disclaimer + response

    def detect_emergency(self, message: str) -> bool:
        """
        Detects emergency keywords in user message.

        Args:
            message: User's input message

        Returns:
            True if emergency detected, False otherwise
        """
        message_lower = message.lower()
        for keyword in self.EMERGENCY_KEYWORDS_AR:
            if keyword in message_lower:
                logger.warning(f"🚨 Emergency keyword detected: {keyword}")
                return True
        return False

    def get_emergency_response(self, language: str = "ar") -> str:
        """
        Returns emergency escalation message.

        Args:
            language: Language code ('ar' or 'en')

        Returns:
            Emergency response text
        """
        return self.EMERGENCY_RESPONSE_AR

    def check_harmful_content(self, message: str) -> Dict[str, Any]:
        """
        Checks for harmful or inappropriate content.

        Args:
            message: User's input message

        Returns:
            dict with 'allowed' (bool) and 'reason' (str) keys
        """
        message_lower = message.lower()

        for topic in self.HARMFUL_TOPICS:
            if topic in message_lower:
                logger.warning(f"⚠️ Harmful content detected: {topic}")
                return {
                    "allowed": False,
                    "reason": "harmful_content",
                    "message": (
                        "عذرًا، لا يمكنني مساعدتك في هذا الموضوع. "
                        "إذا كنت تواجه أزمة، يرجى الاتصال بخط المساعدة النفسية: 920033360"
                    )
                }

        return {"allowed": True, "reason": None}

    def check_turn_limit(self, session_id: str, turn_count: int) -> Dict[str, Any]:
        """
        Enforces maximum conversation turns per session.

        Args:
            session_id: Unique session identifier
            turn_count: Current number of turns in session

        Returns:
            dict with 'allowed' (bool) and 'reason' (str) keys
        """
        if turn_count > self.MAX_TURNS_PER_SESSION:
            logger.warning(
                f"⚠️ Session {session_id} exceeded turn limit: {turn_count}/{self.MAX_TURNS_PER_SESSION}"
            )
            return {
                "allowed": False,
                "reason": "max_turns_exceeded",
                "message": (
                    "لقد تجاوزت الحد الأقصى للمحادثة. "
                    "يرجى بدء جلسة جديدة أو استشارة طبيب مباشرة."
                )
            }

        return {"allowed": True, "reason": None}

    def check_message_length(self, message: str) -> Dict[str, Any]:
        """
        Validates message length.

        Args:
            message: User's input message

        Returns:
            dict with 'allowed' (bool) and 'reason' (str) keys
        """
        if len(message) > self.MAX_MESSAGE_LENGTH:
            logger.warning(f"⚠️ Message too long: {len(message)} chars")
            return {
                "allowed": False,
                "reason": "message_too_long",
                "message": f"الرسالة طويلة جدًا. الحد الأقصى {self.MAX_MESSAGE_LENGTH} حرف."
            }

        if len(message.strip()) < 3:
            return {
                "allowed": False,
                "reason": "message_too_short",
                "message": "الرسالة قصيرة جدًا. يرجى تقديم المزيد من التفاصيل."
            }

        return {"allowed": True, "reason": None}

    def rate_limit_check(self, user_id: str, window_seconds: int = 60, max_requests: int = 10) -> Dict[str, Any]:
        """
        Implements sliding window rate limiting.

        Args:
            user_id: Unique user identifier
            window_seconds: Time window in seconds (default: 60)
            max_requests: Maximum requests per window (default: 10)

        Returns:
            dict with 'allowed' (bool), 'remaining' (int), and 'reset_at' (int) keys
        """
        if not self.redis_client:
            # No Redis available, allow all requests
            return {"allowed": True, "remaining": max_requests, "reset_at": None}

        try:
            key = f"ratelimit:{user_id}"
            now = int(datetime.now().timestamp())

            # Add current request with timestamp as score
            self.redis_client.zadd(key, {str(now): now})

            # Remove old requests outside the window
            self.redis_client.zremrangebyscore(key, 0, now - window_seconds)

            # Count requests in current window
            count = self.redis_client.zcard(key)

            # Set expiry on the key
            self.redis_client.expire(key, window_seconds)

            remaining = max(0, max_requests - count)
            reset_at = now + window_seconds

            if count > max_requests:
                logger.warning(f"⚠️ Rate limit exceeded for user {user_id}: {count}/{max_requests}")
                return {
                    "allowed": False,
                    "remaining": 0,
                    "reset_at": reset_at,
                    "message": f"تجاوزت الحد الأقصى للطلبات. يرجى الانتظار حتى {reset_at}"
                }

            return {
                "allowed": True,
                "remaining": remaining,
                "reset_at": reset_at
            }

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail-closed in production (configurable via env)
            fail_open = os.getenv("GUARDRAILS_FAIL_OPEN", "false").lower() == "true"
            if fail_open:
                logger.warning("Rate limit check failed, but GUARDRAILS_FAIL_OPEN=true - allowing request")
                return {"allowed": True, "remaining": max_requests, "reset_at": None}
            else:
                logger.warning("Rate limit check failed - blocking request (fail-closed)")
                return {
                    "allowed": False,
                    "remaining": 0,
                    "reset_at": None,
                    "message": "خطأ في النظام. يرجى المحاولة لاحقاً."
                }

    def validate_request(
        self,
        message: str,
        user_id: str,
        session_id: str,
        turn_count: int
    ) -> Dict[str, Any]:
        """
        Runs all policy checks on incoming request.

        Args:
            message: User's input message
            user_id: Unique user identifier
            session_id: Unique session identifier
            turn_count: Current turn count in session

        Returns:
            dict with validation results:
            - allowed (bool): Whether request should be processed
            - reason (str): Reason for blocking (if blocked)
            - message (str): User-facing message (if blocked)
            - is_emergency (bool): Whether emergency detected
            - should_add_disclaimer (bool): Whether to add medical disclaimer
        """
        # Check message length
        length_check = self.check_message_length(message)
        if not length_check["allowed"]:
            return {
                "allowed": False,
                "reason": length_check["reason"],
                "message": length_check["message"],
                "is_emergency": False,
                "should_add_disclaimer": False
            }

        # Check rate limit
        rate_check = self.rate_limit_check(user_id)
        if not rate_check["allowed"]:
            return {
                "allowed": False,
                "reason": "rate_limit_exceeded",
                "message": rate_check["message"],
                "remaining": rate_check["remaining"],
                "reset_at": rate_check["reset_at"],
                "is_emergency": False,
                "should_add_disclaimer": False
            }

        # Check turn limit
        turn_check = self.check_turn_limit(session_id, turn_count)
        if not turn_check["allowed"]:
            return {
                "allowed": False,
                "reason": turn_check["reason"],
                "message": turn_check["message"],
                "is_emergency": False,
                "should_add_disclaimer": False
            }

        # Check harmful content
        content_check = self.check_harmful_content(message)
        if not content_check["allowed"]:
            return {
                "allowed": False,
                "reason": content_check["reason"],
                "message": content_check["message"],
                "is_emergency": False,
                "should_add_disclaimer": False
            }

        # Check for emergency
        is_emergency = self.detect_emergency(message)

        logger.info(
            f"✅ Request validated: user={user_id}, session={session_id}, "
            f"turn={turn_count}, emergency={is_emergency}"
        )

        return {
            "allowed": True,
            "reason": None,
            "is_emergency": is_emergency,
            "should_add_disclaimer": True,
            "rate_limit_remaining": rate_check.get("remaining")
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    guardrails = MedicalGuardrails()

    # Test cases
    test_cases = [
        ("عندي صداع منذ يومين", "user-123", "session-1", 1),
        ("أعاني من نوبة قلبية", "user-123", "session-1", 2),
        ("كيف يمكنني الانتحار", "user-456", "session-2", 1),
        ("أريد حجز موعد", "user-123", "session-1", 3),
    ]

    for message, user_id, session_id, turn in test_cases:
        print(f"\n{'='*60}")
        print(f"Message: {message}")
        result = guardrails.validate_request(message, user_id, session_id, turn)
        print(f"Result: {result}")

        if result["allowed"]:
            response = "هذه معلومة طبية مفيدة..."
            if result["should_add_disclaimer"]:
                response = guardrails.inject_disclaimer(response)
            if result["is_emergency"]:
                response = guardrails.get_emergency_response() + response
            print(f"\nFinal Response:\n{response}")
