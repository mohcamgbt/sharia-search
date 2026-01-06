import os
import json
import asyncio
import re
import time
from typing import AsyncGenerator
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import google.generativeai as genai

# ==========================================
# ⚙️ الإعدادات (نفس إعداداتك المستقرة)
# ==========================================
GOOGLE_API_KEY = "AIzaSyDvdlS0Zzl3GA1W8gBICf6S1YmYVP-r4g8"
genai.configure(api_key=GOOGLE_API_KEY)

# الروابط
SCOUT_BASE_URL = "https://f7njguw70xmxr3-8000.proxy.runpod.net/v1" 
SCOUT_API_KEY = "EMPTY"
SCOUT_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-AWQ"

JUDGE_BASE_URL = "https://f7njguw70xmxr3-8000.proxy.runpod.net/v1" 
JUDGE_API_KEY = "EMPTY"
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-AWQ"

REFINER_MODEL_NAME = 'gemini-2.5-pro' # تم التحديث للأسرع والأحدث حسب المتاح

CONTEXT_NEIGHBORS = 2

# ==========================================
# 📝 التلقينات
# ==========================================
STAGE0_PROMPT = """ 
أنت خبير في المصطلحات الفقهية واللغوية.
سؤال المستخدم: "{query}"

المطلوب:
استخرج قائمة بـ 5 إلى 10 كلمات مفتاحية أو مرادفات أو مصطلحات فقهية ذات صلة وثيقة جداً بهذا السؤال، والتي من المحتمل أن توجد في كتب التراث .
- لا تجب على السؤال.
- فقط اكتب الكلمات المفتاحية مفصولة بفواصل.
مثال: لو السؤال "حكم الدخان"، الكلمات: (التبغ، التنباك، شرب الدخان، الخبائث، مفتر).
"""

# تم فصل تعليمات النظام عن المحتوى ليفهمها الموديل بشكل صحيح
STAGE1_SYSTEM_INSTRUCTION = """
You are a relevance analyzer.
You will receive a chunk of text and a query (or keywords).
Your task is to determine if the text is relevant.
Answer ONLY with TRUE or FALSE.
"""

STAGE1_USER_TEMPLATE = """
Context:
"{chunk_text}"

Query: "{query}"

Is the text above related to the query?
If the text mentions, discusses, answers or implies anything about the query (directly, indirectly, or tangentially), answer TRUE.
"""

STAGE1_KEYWORDS_TEMPLATE = """
Context:
"{chunk_text}"

Keywords: [{keywords}]

Does the text above contain these keywords?
Answer ONLY with TRUE or FALSE.
"""


STAGE2_LOCAL_PROMPT = """
<|im_start|>system
You are a critical sniper. 
The provided text contains multiple page markers (e.g., (1/50), (1/51)).
The answer might be in ONE page only, even if the text has 5 pages.

Task: Extract ONLY the specific page numbers that explicitly contain the answer.
RULES:
1. Do NOT list all pages found in the text.
2. If the answer is contained within a single page, return ONLY that page.
3. Only return multiple pages if the answer implies a sentence spanning across them.
4. Be very selective.

Output JSON format: {{"found": true, "pages": ["(1/50)"]}}
<|im_end|>
<|im_start|>user
Question: "{query}"
Text:
"{chunk_text}"
<|im_end|>
<|im_start|>assistant
"""

STAGE3_PROMPT = """
أنت باحث ومحقق في كتاب "زاد المعاد لابن القيم".
السؤال: "{query}"

إليك النصوص المقتبسة من الكتاب:
{context}

التعليمات الصارمة:
1. أجب عن السؤال إجابة وافية من خلال كلام المؤلف فقط ولا تستنبط من عندك شيئا، لا بأس ان تقول مثلا (وذكر كذا، وقال كذا)
2. **التوثيق:** وثق كل معلومة بكتابة رقم صفحتها مباشرة بين قوسين، مثل: **(1/50)** أو **(3/120) وإذا كانت أثر من صفحة فاكتب كل واحدة على حدة ولا تكتبها هكذا **(3/120, 3/121)**.
3. يجب أن يكون رقم الصفحة مطابقاً تماماً لما هو موجود في النصوص أعلاه.
4. اذكر السياق إذا كان مفيداً، حيث أن لديك الصفحات السابقة واللاحقة.
5. لا تخبر المستخدم ان لديك صفحات محددة بل اجب عليه وانت تتظاهر انك قرأت زاد العاد كاملا
"""

# ==========================================
# 🚀 النظام
# ==========================================
class WebSearchSystem:
    def __init__(self, file_path: str):
        self.chunk_size = 8000   
        self.overlap = 1000      
        self.file_path = file_path
        
        self.full_text = ""
        self.chunks_data = [] 
        self.ordered_pages = []
        self.page_label_to_index = {}
        
        self._load_data_into_ram()

        self.client_scout = AsyncOpenAI(api_key=SCOUT_API_KEY, base_url=SCOUT_BASE_URL, timeout=300.0)
        self.client_judge = AsyncOpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL, timeout=300.0)

    def _load_data_into_ram(self):
        print(f"\n📥 [النظام] جاري تحميل الكتاب: {self.file_path} إلى الذاكرة...")
        if not os.path.exists(self.file_path):
            print("❌ الكتاب غير موجود!")
            return
            
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.full_text = f.read()
        
        matches = list(re.finditer(r'\(\d+/\d+\)', self.full_text))
        for i, match in enumerate(matches):
            label = match.group()
            start = match.start()
            end = matches[i+1].start() if i+1 < len(matches) else len(self.full_text)
            self.ordered_pages.append({"label": label, "text": self.full_text[start:end]})
            self.page_label_to_index[label] = i

        step = self.chunk_size - self.overlap
        for i in range(0, len(self.full_text), step):
            chunk = self.full_text[i : i + self.chunk_size]
            self.chunks_data.append({"id": len(self.chunks_data), "text": chunk})
            
        print(f"✅ [النظام] اكتمل التحميل: {len(self.chunks_data)} كاش جاهز في الرام.\n")

    async def emit(self, type_str, content, title=""):
        data = json.dumps({"type": type_str, "content": content, "title": title}, ensure_ascii=False)
        return f"{data}\n"

    # --- المهام الفرعية ---
    async def _scout_task(self, chunk, prompt_content, task_type):
        max_retries = 3      # عدد المحاولات قبل الاستسلام
        base_delay = 2       # مدة الانتظار (ثواني)

        for attempt in range(max_retries):
            try:
                res = await self.client_scout.chat.completions.create(
                    model=SCOUT_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": STAGE1_SYSTEM_INSTRUCTION},
                        {"role": "user", "content": prompt_content}
                    ],
                    max_tokens=1,
                    temperature=0.0,
                    timeout=30.0 # مهلة قصيرة للطلب الواحد
                )
                
                # التحقق من أن الرد ليس صفحة HTML (خطأ كلاودفلير)
                response_text = res.choices[0].message.content.strip().upper()
                
                # إذا رجع لنا HTML بدل النص، نعتبره خطأ ونعيد المحاولة
                if "<!DOCTYPE HTML>" in response_text or "TIMEOUT" in response_text:
                    raise Exception("Cloudflare Timeout HTML Received")

                clean_text = response_text.replace(".", "")
                
                if "TRUE" in clean_text:
                    print(f"🟢 [الكشاف] تم التقاط الكاش #{chunk['id']} | السبب: {task_type}")
                    return (chunk, task_type)
                
                return None # إذا كان FALSE نخرج بسلام

            except Exception as e:
                # طباعة الخطأ والانتظار قليلاً
                print(f"⚠️ [محاولة {attempt+1}/{max_retries}] فشل الكاش #{chunk['id']}: {str(e)[:50]}...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (attempt + 1)) # انتظار تصاعدي (Backoff)
                else:
                    print(f"🔴 [فشل نهائي] تم تجاوز المحاولات للكاش #{chunk['id']}")
                    
        return None



    async def _judge_task(self, chunk, query, caught_by):
        print(f"⚖️ [القاضي] جاري فحص كاش #{chunk['id']} (المصدر: {caught_by})...")
        prompt = STAGE2_LOCAL_PROMPT.format(query=query, chunk_text=chunk['text'])
        try:
            res = await self.client_judge.chat.completions.create(
                model=JUDGE_MODEL_NAME, messages=[{"role": "user", "content": prompt}], max_tokens=350, temperature=0.0, response_format={"type": "json_object"}
            )
            clean = res.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)
            if data.get("found") is True and data.get("pages"):
                pages = data['pages']
                if len(pages) > 0:
                    best_page = [pages[0]]
                    print(f"✅ [القاضي] اعتمد الصفحة: {best_page}")
                    return best_page
            else:
                print(f"🗑️ [القاضي] لم يجد إجابة في كاش #{chunk['id']}")
        except Exception as e:
            print(f"💥 [خطأ القاضي] {e}")
        return []

    # --- المعالج الرئيسي ---
    async def process_stream(self, query: str) -> AsyncGenerator[str, None]:
        print("\n" + "="*60 + f"\n🚀 [بدء طلب جديد] السؤال: {query}\n" + "="*60)

        # 1. التخطيط
        yield await self.emit("evidence", "تحليل السؤال...", "المخطط")
        keywords = query
        try:
            print("🤖 [جيميني] جاري استخراج الكلمات المفتاحية...")
            model = genai.GenerativeModel(REFINER_MODEL_NAME)
            resp = await model.generate_content_async(STAGE0_PROMPT.format(query=query))
            
            # --- إصلاح الخطأ 1: التحقق من وجود نص قبل الوصول إليه ---
            if resp.parts:
                keywords = resp.text.strip()
                print(f"🔑 [الكلمات المفتاحية المستخرجة]: {keywords}")
                yield await self.emit("evidence", f"الكلمات: {keywords}", "Gemini")
            else:
                print("⚠️ [تنبيه] جيميني لم يرجع نصاً للكلمات المفتاحية.")
                keywords = query

        except Exception as e: 
            print(f"⚠️ [تنبيه] فشل استخراج الكلمات المفتاحية ({e})، سنستخدم السؤال فقط.")
            keywords = query

        # 2. المسح
        yield await self.emit("evidence", "المسح السريع (Turbo Mode)...", "Scout Launch")
        sem_scout = asyncio.Semaphore(500) 

        async def protected_scout(chunk, prompt, type):
            async with sem_scout:
                return await self._scout_task(chunk, prompt, type)

        tasks = []
        for chunk in self.chunks_data:
            p1 = STAGE1_USER_TEMPLATE.format(query=query, chunk_text=chunk['text'])
            tasks.append(protected_scout(chunk, p1, "تطابق السؤال"))
            p2 = STAGE1_KEYWORDS_TEMPLATE.format(keywords=keywords, chunk_text=chunk['text'])
            tasks.append(protected_scout(chunk, p2, "تطابق الكلمات المفتاحية"))
        
        results = await asyncio.gather(*tasks)
        
        hits = {}
        for r in results:
            if r:
                chunk, reason = r
                if chunk['id'] in hits:
                    prev_chunk, prev_reason = hits[chunk['id']]
                    if reason not in prev_reason:
                        hits[chunk['id']] = (chunk, f"{prev_reason} + {reason}")
                else:
                    hits[chunk['id']] = (chunk, reason)

        suspicious_items = list(hits.values())
        
        if not suspicious_items:
            print("🛑 [توقف] لم يجد الكشاف أي نتيجة.")
            yield await self.emit("token", "لم أجد نتائج في الكتاب حول هذا الموضوع.")
            return

        print(f"📊 [ملخص الكشاف] تم الاشتباه في {len(suspicious_items)} كاش.")
        yield await self.emit("evidence", f"اشتباه في {len(suspicious_items)} موضع...", "Summary")

        # 3. التدقيق
        yield await self.emit("evidence", "تدقيق الصفحات...", "Judge")
        sem_judge = asyncio.Semaphore(50)
        async def protected_judge(item):
            chunk, reason = item
            async with sem_judge: 
                return await self._judge_task(chunk, query, reason)

        judge_tasks = [protected_judge(item) for item in suspicious_items]
        judge_results = await asyncio.gather(*judge_tasks)
        
        valid_pages = sorted(list(set([p for sub in judge_results if sub for p in sub])))
        print(f"👑 [النتيجة النهائية] الصفحات المعتمدة: {valid_pages}")
        
        if not valid_pages:
            yield await self.emit("token", "لم أجد إجابة دقيقة في الصفحات التي اشتبهت بها.")
            return

        # 4. بناء السياق
        yield await self.emit("evidence", "تجهيز السياق...", "Context")
        
        context_str = ""
        references_map = {} 
        gemini_reading_log = []

        for label in valid_pages:
            clean_label = label.strip()
            found_idx = -1
            
            if clean_label in self.page_label_to_index: found_idx = self.page_label_to_index[clean_label]
            elif f"({clean_label})" in self.page_label_to_index: found_idx = self.page_label_to_index[f"({clean_label})"]
            else:
                for key in self.page_label_to_index:
                    if key.replace(" ", "") == clean_label.replace(" ", ""):
                        found_idx = self.page_label_to_index[key]; clean_label = key; break
            
            if found_idx != -1:
                start_idx = max(0, found_idx - CONTEXT_NEIGHBORS)
                end_idx = min(len(self.ordered_pages), found_idx + CONTEXT_NEIGHBORS + 1)
                
                context_str += f"\n--- سياق الصفحة المستهدفة {clean_label} ---\n"
                pages_bundle = []
                
                for ctx_i in range(start_idx, end_idx):
                    page_obj = self.ordered_pages[ctx_i]
                    is_target = (ctx_i == found_idx)
                    marker = " (الصفحة المستهدفة)" if is_target else ""
                    
                    if page_obj['label'] not in gemini_reading_log:
                        context_str += f"صفحة {page_obj['label']}{marker}:\n{page_obj['text']}\n"
                        gemini_reading_log.append(page_obj['label'])
                    
                    pages_bundle.append({
                        "label": page_obj['label'],
                        "text": page_obj['text']
                    })
                
                context_str += "-" * 20 + "\n"
                
                # إنشاء الحزم للمراجع
                for focus_item in pages_bundle:
                    custom_bundle = []
                    for p in pages_bundle:
                        custom_bundle.append({
                            "label": p['label'],
                            "text": p['text'],
                            "is_target": (p['label'] == focus_item['label'])
                        })
                    references_map[focus_item['label']] = custom_bundle

        # 5. الصياغة
        final_prompt = STAGE3_PROMPT.format(query=query, context=context_str)
        try:
            print("✍️ [الصائغ] جاري كتابة الإجابة...")
            model_pro = genai.GenerativeModel(REFINER_MODEL_NAME)
            stream = await model_pro.generate_content_async(final_prompt, stream=True)
            
            async for chunk in stream:
                # --- إصلاح الخطأ 2: الوصول الآمن للنص داخل التكرار ---
                try:
                    if chunk.parts:
                        text_part = chunk.text
                        if text_part:
                            yield await self.emit("token", text_part)
                except Exception:
                    continue # تجاهل الأجزاء الفارغة
            
            refs_json = json.dumps(references_map, ensure_ascii=False)
            yield await self.emit("token", f"\n\n---REFERENCES_START---{refs_json}---REFERENCES_END---")
            print("✅ [تم] اكتمل الطلب بنجاح.")
            
        except Exception as e:
            print(f"❌ [خطأ الصائغ] {e}")
            yield await self.emit("token", f" حدث خطأ أثناء الصياغة: {str(e)}")


# ==========================================
# 🚀 السيرفر
# ==========================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

system_instance = WebSearchSystem("book2.txt")

@app.post("/chat_stream")
async def chat_endpoint(request: Request):
    data = await request.json()
    return StreamingResponse(system_instance.process_stream(data.get("message", "")), media_type="application/x-ndjson")