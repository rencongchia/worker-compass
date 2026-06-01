"""
All prompt templates and multilingual static strings for WorkerCompass.

Static refusal and disclaimer strings were translated by Qwen3-32B and are
flagged for native speaker review before production use.
"""

# ---------------------------------------------------------------------------
# Language metadata
# ---------------------------------------------------------------------------

LANGUAGES = {
    "en": {"name": "English",   "display": "English"},
    "bn": {"name": "Bengali",   "display": "বাংলা"},
    "ta": {"name": "Tamil",     "display": "தமிழ்"},
    "my": {"name": "Burmese",   "display": "မြန်မာ"},
}

LANGUAGE_NAMES = {code: meta["name"] for code, meta in LANGUAGES.items()}

# ---------------------------------------------------------------------------
# Translation prompt (non-English → English)
# ~50 input / ~30 output tokens; add ~200ms latency
# ---------------------------------------------------------------------------

TRANSLATION_PROMPT = """\
Translate the following text to English. Preserve all legal terms, monetary amounts,
timeframes, proper nouns, and Singapore-specific acronyms exactly (e.g. TADM, WICA,
ECT, MOM, TWC2). Output ONLY the translated English text — no preamble, no explanation.

Text to translate:
{query}"""

# ---------------------------------------------------------------------------
# Generation system prompt
# ---------------------------------------------------------------------------

GENERATION_SYSTEM = """\
You are a legal information assistant helping low-wage migrant workers in Singapore
understand their employment rights.

Rules you must follow:
1. Answer ONLY using information from the provided legal documents. Do not add any
   information not present in those documents.
2. Use superscript citations (¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸) SPARINGLY — maximum 3 per response, and
   ONLY when stating a specific legal threshold, deadline, or monetary limit taken directly
   from a document (e.g. "within 7 days²", "$500 per day³"). Do NOT cite after general
   statements, summaries, or procedural steps.
3. Answer in {language_name}. If the user wrote in {language_name}, reply in {language_name}.
4. Be practical: explain what the worker can actually DO, not just what the law says.
5. If the documents do not fully answer the question, say so explicitly — do not guess.
6. Keep your answer concise. Use bullet points for step-by-step procedures.
7. Never provide specific legal advice about the user's individual case. Direct them
   to MOM (6438 5122) or TWC2 for case-specific guidance.\
"""

GENERATION_USER = """\
{history_block}The worker asked (in {language_name}):
{original_query}

Relevant legal documents:
{chunks}

Answer the latest question in {language_name} based only on the documents above.
Cite sparingly: use a superscript only for a specific legal number, deadline, or monetary limit (e.g. "within 14 days¹"). Do not add a citation after every sentence.\
"""

# Format a single chunk for insertion into the generation prompt
CHUNK_TEMPLATE = "{sup} {act_name} — {section_title}\nURL: {url}\n\n{text}"

# ---------------------------------------------------------------------------
# Refusal messages (translated by Qwen3; flag for native speaker review)
# ---------------------------------------------------------------------------

REFUSALS = {
    "en": (
        "I cannot find information about this topic in my knowledge base. "
        "For help with this question, please:\n"
        "- Call MOM at **6438 5122** (Mon–Fri, 8:30am–5:30pm)\n"
        "- Contact **TWC2** at twc2.org.sg\n"
        "- Visit a **Social Service Office** or the **Migrant Workers' Centre (MWC)**"
    ),
    "bn": (
        "আমি আমার জ্ঞানভাণ্ডারে এই বিষয়ে তথ্য খুঁজে পাচ্ছি না। "
        "এই প্রশ্নের সাহায্যের জন্য অনুগ্রহ করে:\n"
        "- MOM-কে **6438 5122** নম্বরে কল করুন (সোম–শুক্র, সকাল ৮:৩০–বিকাল ৫:৩০)\n"
        "- **TWC2**-এ যোগাযোগ করুন: twc2.org.sg\n"
        "- একটি **সামাজিক সেবা অফিস** বা **Migrant Workers' Centre (MWC)**-এ যান"
    ),
    "ta": (
        "என்னுடைய அறிவுத் தளத்தில் இந்த தலைப்பில் தகவல்களை என்னால் கண்டுபிடிக்க முடியவில்லை. "
        "இந்த கேள்விக்கு உதவிக்கு:\n"
        "- MOM-ஐ **6438 5122** என்ற எண்ணில் அழைக்கவும் (திங்கள்–வெள்ளி, காலை 8:30–மாலை 5:30)\n"
        "- **TWC2**-ஐ தொடர்பு கொள்ளுங்கள்: twc2.org.sg\n"
        "- ஒரு **Social Service Office** அல்லது **Migrant Workers' Centre (MWC)**-ஐ பார்வையிடுங்கள்"
    ),
    "my": (
        "ကျွန်ုပ်၏ အသိပညာအခြေခံတွင် ဤအကြောင်းအရာနှင့် ပတ်သက်သော အချက်အလက်ကို ရှာမတွေ့ပါ။ "
        "ဤမေးခွန်းနှင့် ပတ်သတ်သည့် အကူအညီအတွက်:\n"
        "- MOM ကို **6438 5122** သို့ ဖုန်းဆက်ပါ (တနင်္လာ–သောကြာ၊ နံနက် ၈:၃၀–ညနေ ၅:၃၀)\n"
        "- **TWC2** ကို twc2.org.sg တွင် ဆက်သွယ်ပါ\n"
        "- **Social Service Office** သို့မဟုတ် **Migrant Workers' Centre (MWC)** သို့ သွားပါ"
    ),
}

# ---------------------------------------------------------------------------
# Disclaimer (appended to every generated answer)
# ---------------------------------------------------------------------------

DISCLAIMERS = {
    "en": (
        "*This information is general guidance only and is not legal advice. "
        "For your specific situation, contact MOM (6438 5122), TWC2, or a legal aid organisation.*"
    ),
    "bn": (
        "*এই তথ্য শুধুমাত্র সাধারণ নির্দেশনার জন্য এবং এটি আইনি পরামর্শ নয়। "
        "আপনার নির্দিষ্ট পরিস্থিতির জন্য MOM (6438 5122), TWC2 বা আইনি সহায়তা সংস্থার সাথে যোগাযোগ করুন।*"
    ),
    "ta": (
        "*இந்தத் தகவல் பொதுவான வழிகாட்டுதலுக்காக மட்டுமே, இது சட்ட ஆலோசனை அல்ல. "
        "உங்கள் குறிப்பிட்ட சூழ்நிலைக்கு MOM (6438 5122), TWC2 அல்லது சட்ட உதவி நிறுவனத்தை தொடர்பு கொள்ளுங்கள்.*"
    ),
    "my": (
        "*ဤအချက်အလက်သည် ယေဘုယျလမ်းညွှန်မှုအတွက်သာဖြစ်ပြီး ဥပဒေဆိုင်ရာ အကြံဉာဏ်မဟုတ်ပါ။ "
        "သင်၏ သီးခြားအခြေအနေအတွက် MOM (6438 5122)၊ TWC2 သို့မဟုတ် ဥပဒေအကူအညီ အဖွဲ့အစည်းကို ဆက်သွယ်ပါ။*"
    ),
}

# ---------------------------------------------------------------------------
# UI placeholder text for the query input box, per language
# ---------------------------------------------------------------------------

PLACEHOLDERS = {
    "en": "e.g. My employer has not paid my salary for 2 months. What can I do?",
    "bn": "যেমন: আমার নিয়োগকর্তা ২ মাস ধরে বেতন দেননি। আমি কী করতে পারি?",
    "ta": "எ.கா: என் முதலாளி 2 மாதமாக சம்பளம் கொடுக்கவில்லை. நான் என்ன செய்யலாம்?",
    "my": "ဥပမာ - ကျွန်ုပ်၏ အလုပ်ရှင်သည် လပေါင်း ၂ လကြာ လစာမပေးပါ။ ကျွန်ုပ် ဘာလုပ်နိုင်သလဲ？",
}

# ---------------------------------------------------------------------------
# Freshness warning (appended when corpus_snapshot_date is stale)
# ---------------------------------------------------------------------------

FRESHNESS_WARNING = {
    "en": "\n\n⚠️ *Note: the source document for this answer was last updated {days} days ago. "
          "Legal limits and procedures may have changed. Verify at mom.gov.sg before acting.*",
    "bn": "\n\n⚠️ *দ্রষ্টব্য: এই উত্তরের উৎস নথিটি {days} দিন আগে সর্বশেষ আপডেট হয়েছিল।*",
    "ta": "\n\n⚠️ *குறிப்பு: இந்த பதிலுக்கான ஆதார ஆவணம் {days} நாட்களுக்கு முன்பு புதுப்பிக்கப்பட்டது.*",
    "my": "\n\n⚠️ *မှတ်ချက် - ဤဖြေဆိုချက်၏ မူလအရင်းအမြစ်ကို ရက်ပေါင်း {days} ကြာ မပြောင်းလဲမီ သတ်မှတ်ထားသည်။*",
}
