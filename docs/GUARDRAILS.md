# 🛡️ Medical Policy Guardrails

## Overview

The Medical Guardrails system enforces safety policies, content filtering, and rate limiting for HealthTech AI medical conversations. It ensures compliance with medical ethics and protects both users and the organization.

## Features

### 1. **Medical Disclaimers** 📋
- Automatically adds disclaimers to all AI responses
- Bilingual support (Arabic & English)
- Clear notice that AI is not a replacement for human doctors

### 2. **Emergency Detection** 🚨
- Real-time detection of emergency keywords
- Immediate escalation for critical symptoms
- Emergency contact information display
- Keywords monitored:
  - نوبة قلبية (heart attack)
  - صعوبة تنفس (difficulty breathing)
  - نزيف شديد (severe bleeding)
  - فقدان وعي (loss of consciousness)
  - جلطة/سكتة (stroke)

### 3. **Harmful Content Blocking** ⚠️
- Filters harmful or inappropriate topics
- Crisis hotline referrals for mental health
- Protected topics:
  - Self-harm discussions
  - Suicide-related content
  - Illegal drug information
  - Dangerous medical advice

### 4. **Rate Limiting** ⏱️
- Prevents system abuse
- Sliding window algorithm
- Default: 10 requests per minute per user
- Redis-backed for distributed environments
- Graceful degradation if Redis unavailable

### 5. **Session Turn Limiting** 🔄
- Maximum 20 turns per conversation session
- Encourages users to seek professional help
- Prevents over-reliance on AI

### 6. **Message Validation** ✅
- Minimum 3 characters
- Maximum 2000 characters
- Prevents spam and abuse

## Architecture

```
User Message
     ↓
┌─────────────────────────────────────┐
│   MedicalGuardrails.validate_request │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  1. Check Message Length            │
│  2. Check Rate Limit (Redis)        │
│  3. Check Turn Limit                │
│  4. Check Harmful Content           │
│  5. Detect Emergency                │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Return Validation Result:          │
│  - allowed (bool)                   │
│  - reason (str)                     │
│  - is_emergency (bool)              │
│  - should_add_disclaimer (bool)     │
└─────────────────────────────────────┘
     ↓
AI Response Processing
     ↓
┌─────────────────────────────────────┐
│  MedicalGuardrails.inject_disclaimer│
└─────────────────────────────────────┘
     ↓
Final Response to User
```

## Usage

### Basic Example

```python
from guardrails import MedicalGuardrails
import redis

# Initialize with Redis (optional)
redis_client = redis.Redis(host='localhost', port=6379, db=0)
guardrails = MedicalGuardrails(redis_client=redis_client)

# Validate incoming request
result = guardrails.validate_request(
    message="عندي صداع منذ يومين",
    user_id="user-123",
    session_id="session-456",
    turn_count=5
)

if not result["allowed"]:
    # Request blocked
    return {"error": result["message"], "reason": result["reason"]}

if result["is_emergency"]:
    # Emergency detected - escalate immediately
    emergency_response = guardrails.get_emergency_response()
    return {"response": emergency_response, "escalate": True}

# Process with LLM
ai_response = llm.generate(message)

# Add disclaimer
if result["should_add_disclaimer"]:
    ai_response = guardrails.inject_disclaimer(ai_response, language="ar")

return {"response": ai_response}
```

### Integration with Gateway (NestJS)

```typescript
// conversation.service.ts
import { Injectable } from '@nestjs/common';
import { execSync } from 'child_process';

@Injectable()
export class ConversationService {
  async sendMessage(dto: SendMessageDto) {
    // Call guardrails via Python subprocess
    const guardrailsResult = execSync(
      `python guardrails_check.py "${dto.message}" "${dto.userId}" "${dto.sessionId}" ${dto.turnCount}`
    ).toString();
    
    const validation = JSON.parse(guardrailsResult);
    
    if (!validation.allowed) {
      throw new BadRequestException(validation.message);
    }
    
    if (validation.is_emergency) {
      // Escalate to emergency protocol
      return this.escalateEmergency(dto);
    }
    
    // Continue normal processing
    const response = await this.llmService.generate(dto.message);
    
    // Add disclaimer
    if (validation.should_add_disclaimer) {
      response.text = this.addDisclaimer(response.text);
    }
    
    return response;
  }
}
```

## Configuration

### Environment Variables

```bash
# Redis Configuration (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Rate Limiting
RATE_LIMIT_WINDOW_SECONDS=60
RATE_LIMIT_MAX_REQUESTS=10

# Session Limits
MAX_TURNS_PER_SESSION=20
MAX_MESSAGE_LENGTH=2000
```

### Custom Guardrails

```python
# Create custom guardrails instance
guardrails = MedicalGuardrails(redis_client=redis_client)

# Override defaults
guardrails.MAX_TURNS_PER_SESSION = 30
guardrails.MAX_MESSAGE_LENGTH = 3000

# Add custom emergency keywords
guardrails.EMERGENCY_KEYWORDS_AR.extend([
    "إغماء",  # fainting
    "تشنجات",  # seizures
])

# Add custom harmful topics
guardrails.HARMFUL_TOPICS.extend([
    "تعاطي المخدرات",
    "العنف",
])
```

## Testing

```bash
# Run standalone test
python services/llm/guardrails.py

# Expected output:
# ✅ Request validated: user=user-123, session=session-1, turn=1
# ⚠️ Harmful content detected: انتحار
# 🚨 Emergency keyword detected: نوبة قلبية
```

## Response Examples

### Normal Response (with disclaimer)
```
⚠️ **تنويه طبي مهم**: أنا مساعد ذكي وليس طبيبًا بشريًا. 
المعلومات المقدمة لأغراض تعليمية فقط. يرجى استشارة طبيب مؤهل 
للحصول على تشخيص وعلاج دقيق.

الصداع يمكن أن يكون ناتجًا عن عدة أسباب مثل التوتر، قلة النوم، 
أو الجفاف. أنصحك بشرب الماء والراحة...
```

### Emergency Response
```
🚨 **حالة طارئة محتملة**: تم رصد أعراض قد تكون خطيرة. 
يرجى الاتصال بالإسعاف فورًا على رقم 123 أو التوجه إلى أقرب مستشفى. 
لا تنتظر وابحث عن مساعدة طبية فورية.

📞 أرقام الطوارئ:
- الإسعاف: 123
- الشرطة: 122
- الدفاع المدني: 125
```

### Harmful Content Blocked
```
عذرًا، لا يمكنني مساعدتك في هذا الموضوع. 
إذا كنت تواجه أزمة، يرجى الاتصال بخط المساعدة النفسية: 920033360
```

### Rate Limit Exceeded
```
تجاوزت الحد الأقصى للطلبات. يرجى الانتظار حتى 1736789520
```

## Metrics & Monitoring

### Prometheus Metrics (Recommended)

```python
from prometheus_client import Counter, Histogram

# Add metrics
guardrails_requests_total = Counter(
    'guardrails_requests_total',
    'Total guardrails validation requests',
    ['result']
)

guardrails_blocked_total = Counter(
    'guardrails_blocked_total',
    'Total blocked requests',
    ['reason']
)

guardrails_emergency_detected_total = Counter(
    'guardrails_emergency_detected_total',
    'Total emergency detections'
)

# Instrument the code
result = guardrails.validate_request(...)
guardrails_requests_total.labels(result='allowed' if result['allowed'] else 'blocked').inc()

if not result['allowed']:
    guardrails_blocked_total.labels(reason=result['reason']).inc()

if result['is_emergency']:
    guardrails_emergency_detected_total.inc()
```

## Production Checklist

- [ ] Redis configured for rate limiting
- [ ] Emergency phone numbers updated for region
- [ ] Crisis hotline numbers configured
- [ ] Prometheus metrics integrated
- [ ] Alerting configured for blocked content
- [ ] Logging configured for audit trail
- [ ] Language-specific disclaimers reviewed by legal
- [ ] Emergency keywords reviewed by medical staff
- [ ] Rate limits tuned based on usage patterns

## Future Enhancements

1. **ML-based Content Filtering**
   - Replace keyword matching with ML model
   - Better detection of harmful content
   - Reduced false positives

2. **Contextual Disclaimers**
   - Different disclaimers for different medical topics
   - Severity-based disclaimer levels

3. **Multi-language Support**
   - Add more languages
   - Region-specific emergency numbers

4. **Advanced Rate Limiting**
   - User-tier based limits
   - Dynamic limits based on behavior
   - Per-endpoint limits

5. **Audit Logging**
   - Log all blocked requests
   - Compliance reporting
   - GDPR-compliant storage

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
- GitHub Issues: [github.com/yourorg/healthtech-ai](https://github.com)
- Email: support@healthtech-ai.com
- Docs: [docs.healthtech-ai.com](https://docs.healthtech-ai.com)
