import os
import torch
from TTS.api import TTS

# -------- CONFIG --------
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SPEAKER_NAME = "Vjollca Johnnie"   # <-- this is the voice you liked
LANGUAGE = "ar"
OUT_DIR = "xtts_suad_medical_samples"

# Arabic medical phrases for a hospital voice agent
MEDICAL_PHRASES = [
    "مرحباً، أنت الآن تتحدث مع المساعد الصوتي للنظام الصحي.",
    "كيف أستطيع مساعدتك اليوم؟ هل تعاني من عرض جديد أو تريد الاستفسار عن موعد؟",
    "قبل أن نبدأ، من فضلك لا تذكر رقم الهوية أو أي معلومات حساسة جداً خلال المكالمة.",
    "هل تتصل عن نفسك أم عن مريض آخر مثل أحد أفراد العائلة؟",
    "هل تعاني حالياً من ألم في الصدر، أو صعوبة شديدة في التنفس، أو نزيف لا يتوقف؟",
    "إذا كانت هذه حالة طارئة، يرجى التوجه فوراً إلى أقرب قسم طوارئ أو الاتصال بالإسعاف.",
    "سأسألك بعض الأسئلة القصيرة لأفهم حالتك بشكل أفضل.",
    "ما هو العرض الرئيسي الذي تعاني منه الآن؟ على سبيل المثال صداع، حرارة، كحة، أو ألم في البطن.",
    "منذ متى بدأت الأعراض التي تشعر بها؟",
    "هل لديك أمراض مزمنة مثل السكري، ضغط الدم المرتفع، أو أمراض القلب؟",
    "هل تتناول أي أدوية بشكل منتظم؟ إذا نعم، حاول أن تذكر أسماءها أو نوعها.",
    "هل لديك حساسية معروفة من أي دواء أو أطعمة أو مواد معينة؟",
    "هل قمت بقياس درجة الحرارة أو ضغط الدم في البيت؟ وما هي القراءة تقريباً؟",
    "بالاعتماد على المعلومات التي ذكرتها، أنصحك بحجز موعد في العيادة خلال أقرب وقت ممكن.",
    "تم تسجيل طلبك، وسيتواصل معك أحد أفراد الفريق الطبي لتأكيد الموعد والتفاصيل.",
    "تذكّر أن هذه الخدمة لا تغني عن مراجعة الطبيب أو الذهاب للطوارئ في الحالات الخطيرة.",
    "شكراً لاتصالك، نتمنى لك الصحة والسلامة دائماً."
]


def main():
    print("Loading XTTS v2 model... this can take a bit the first time.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load XTTSv2 (uses the already-downloaded model in your TTS_HOME / HF_HOME)
    tts = TTS(MODEL_NAME).to(device)

    os.makedirs(OUT_DIR, exist_ok=True)

    for idx, text in enumerate(MEDICAL_PHRASES, start=1):
        filename = f"{idx:02d}_vjollca_johnnie_medical.wav"
        path = os.path.join(OUT_DIR, filename)

        print(f"Generating {idx}/{len(MEDICAL_PHRASES)} -> {path}")
        tts.tts_to_file(
            text=text,
            speaker=SPEAKER_NAME,
            language=LANGUAGE,
            file_path=path,
        )

    print(f"\n✅ Done. Check the '{OUT_DIR}' folder for vjollca's medical samples.")


if __name__ == "__main__":
    main()
