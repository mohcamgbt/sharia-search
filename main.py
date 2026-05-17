import os
import json
import asyncio
import re
import tempfile
import time
from typing import AsyncGenerator
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
import google.generativeai as genai
import uvicorn

# ==========================================
# ⚙️ الإعدادات 
# ==========================================
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = 'gemini-2.5-pro'
CONTEXT_NEIGHBORS = 2

BOOKS_INFO = [
    {"path": "قرارات الهيئة الشرعية بمصرف الراجحي (الجزء الأول)_vision.txt", "name": "قرارات الهيئة الراجحي (ج1)"},
    {"path": "قرارات الهيئة الشرعية بمصرف الراجحي (الجزء الثاني)_vision.txt", "name": "قرارات الهيئة الراجحي (ج2)"},
    {"path": "المعايير الشرعية ايوفي_vision.txt", "name": "المعايير الشرعية ايوفي"}
]

# ==========================================
# 📝 التلقينات
# ==========================================
ROUTER_PROMPT = """
أنت بواب ذكي لمنصة "حكيم".
إليك سجل المحادثة السابق لفهم السياق (إن وجد):
{history_text}

رسالة المستخدم الحالية: "{query}"

المطلوب:
بناءً على السياق والرسالة الحالية، قم بتحليل القصد واستخراج كود JSON فقط بالصيغة التالية، ولا تكتب أي حرف خارج الـ JSON:
{{
  "action": "CHAT" أو "SEARCH",
  "chat_reply": "ردك الودي هنا إذا كانت الرسالة تحية أو شكر",
  "search_query": "سؤال البحث المستقل تماماً"
}}

تحذير شديد جداً وقواعد هامة:
1. إذا قال المستخدم "السلام عليكم" أو "أهلا" أو شكرك، اجعل action = CHAT ورد برد ودي قصير.
2. إذا كانت الرسالة سؤالاً أو استكمالاً لسؤال سابق، قم بصياغة سؤال مستقل وواضح للبحث واجعله في search_query.
"""

MAP_PROMPT = """
الكتاب الحالي: {book_name}
السؤال الموجه للبحث: "{query}"

مهمتك: قم بإجراء مسح شامل في النص المرفق للبحث عن إجابة.
القواعد الصارمة:
1. إذا لم تجد إجابة، اكتب: EMPTY
2. التوثيق إلزامي ويكون في نهاية الجملة باستخدام أقواس مربعة مزدوجة حصراً هكذا: [[{book_name}، ص رقم_الصفحة]].
3. رقم الصفحة الذي يجب أن تستخدمه في التوثيق موجود في بداية كل صفحة بين قوسين هكذا: [الصفحة: 87]. استخدم هذا الرقم الرقمي فقط وتجاهل الأرقام العربية بأسفل النص.
مثال للتوثيق الصحيح: [[{book_name}، ص 87]].
4. يُمنع منعاً باتاً دمج الصفحات في توثيق واحد مثل [[الكتاب، ص 87-88]]. وثق كل معلومة برقم صفحة واحد فقط.
5. يُمنع كتابة اسم الكتاب في الإجابة بدون توثيق رقم الصفحة والأقواس المزدوجة.
"""

REDUCE_PROMPT = """
السؤال الأصلي: "{query}"

النتائج المستخرجة:
{gathered_info}

المطلوب:
1. صغ إجابة نهائية ومنظمة للمستخدم بناءً على هذه النتائج فقط.
2. التوثيق الصارم: حافظ على التوثيقات كما هي تماماً بالأقواس المربعة المزدوجة: [[اسم الكتاب، ص رقم الصفحة الرقمي]].
مثال: [[القرارات والتوصيات، ص 87]].
3. تحذيرات قاتلة:
- إياك أن تكتب اسم الكتاب هكذا (القرارات والتوصيات) بدون وضعه داخل الأقواس المزدوجة مع رقم الصفحة.
- إياك أن تدمج عدة أرقام صفحات في توثيق واحد مثل [[الكتاب، ص 87، 88]]. اختر رقم صفحة واحد فقط للمعلومة.
- يُمنع تحويل التوثيق إلى أقواس مفردة ().
"""

# ==========================================
# 🚀 النظام
# ==========================================
class WebSearchSystem:
    def __init__(self):
        self.books_data = {} 
        self.last_upload_time = 0
        self.upload_interval = 40 * 3600  # 40 ساعة بالثواني
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        print("\n📥 [النظام] جاري تحميل الكتب واستخراج الأرقام الحقيقية للصفحات...")
        
        # تنظيف الملفات القديمة من سيرفرات جوجل لتجنب تراكمها وتجاوز حدود الحساب
        for b_name, data in self.books_data.items():
            if "uploaded_file" in data:
                try:
                    genai.delete_file(data["uploaded_file"].name)
                    print(f"🧹 تم حذف النسخة القديمة من: {b_name}")
                except Exception as e:
                    print(f"⚠️ لم يتمكن من حذف {b_name} القديم: {e}")
        
        self.books_data.clear()

        for book in BOOKS_INFO:
            path = book["path"]
            b_name = book["name"]
            
            if not os.path.exists(path):
                print(f"⚠️ تنبيه: الكتاب '{path}' غير موجود.")
                continue
                
            with open(path, 'r', encoding='utf-8') as f:
                book_text = f.read()

            print(f"📄 معالجة صفحات كتاب: {b_name}...")
            chunks = re.split(r'\[\-\-\-\s*Page\s*(\d+)\s*\-\-\-\]', book_text)
            
            processed_text = ""
            pages_list = []
            
            if len(chunks) > 1:
                for i in range(1, len(chunks), 2):
                    pdf_page_num = chunks[i].strip()
                    page_text = chunks[i+1].strip()
                    
                    if not page_text:
                        continue

                    lines = [line.strip() for line in page_text.split('\n') if line.strip()]
                    real_page_num = pdf_page_num
                    if lines and len(lines[-1]) <= 15:
                        real_page_num = lines[-1]
                        
                    marked_chunk = f"\n\n[الصفحة: {pdf_page_num}]\n{page_text}"
                    processed_text += marked_chunk
                    
                    pages_list.append({
                        "label": pdf_page_num,        
                        "real_label": real_page_num,  
                        "text": page_text
                    })

            try:
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as tmp:
                    tmp.write(processed_text)
                    tmp_path = tmp.name
                    
                uploaded_file = genai.upload_file(path=tmp_path, display_name=f"حكيم_{b_name}")
                os.remove(tmp_path)
                
                self.books_data[b_name] = {
                    "uploaded_file": uploaded_file,
                    "pages": pages_list
                }
                print(f"✅ تم الرفع! URI: {uploaded_file.uri}")
            except Exception as e:
                print(f"❌ خطأ في رفع {b_name}: {e}")

        # تحديث وقت آخر عملية رفع ناجحة
        self.last_upload_time = time.time()

    async def emit(self, type_str, content, title=""):
        data = json.dumps({"type": type_str, "content": content, "title": title}, ensure_ascii=False)
        return f"{data}\n"

    async def process_stream(self, query: str, history: list) -> AsyncGenerator[str, None]:
        # آلية التجديد التلقائي قبل تجاوز 48 ساعة
        if time.time() - self.last_upload_time > self.upload_interval:
            print("🔄 [النظام] مرت 40 ساعة، جاري تجديد رفع الملفات تلقائياً لتجنب الحذف...")
            yield await self.emit("evidence", "جاري تحديث ملفات النظام والموسوعة لضمان استمرار الخدمة...", "تحديث النظام")
            self._load_and_prepare_data()

        print("\n" + "="*60 + f"\n🚀 [طلب جديد] السؤال الأصلي: {query}\n" + "="*60)
        
        model = genai.GenerativeModel(MODEL_NAME)
        
        # 1. تجهيز السياق
        history_text = ""
        if history:
            for msg in history[-4:]: 
                role = "المستخدم" if msg['role'] == "user" else "حكيم"
                history_text += f"{role}: {msg['content']}\n"
        if not history_text:
            history_text = "لا يوجد سجل سابق (هذه أول رسالة)."

        # 2. الموجه الصارم (Router)
        yield await self.emit("evidence", "تحليل سياق المحادثة وصياغة استعلام البحث...", "الموجه الذكي")
        try:
            router_res = await model.generate_content_async(ROUTER_PROMPT.format(history_text=history_text, query=query))
            raw_text = router_res.text.strip()
            
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                clean_json = json_match.group(0)
                router_data = json.loads(clean_json)
                
                if router_data.get("action") == "CHAT":
                    print("👋 [الموجه] الرسالة محادثة/تحية. تم الرد وإيقاف البحث.")
                    yield await self.emit("token", router_data.get("chat_reply", "أهلاً بك في منصة حكيم!"))
                    return
                else:
                    query = router_data.get("search_query", query)
                    print(f"🔍 [الموجه] استقر سؤال البحث النهائي على: {query}")
                    yield await self.emit("evidence", f"سؤال البحث المعتمد: {query}", "تأكيد البحث")
            else:
                print("⚠️ لم يتم العثور على JSON في رد الموجه، سيتم الإكمال بالسؤال الأصلي.")
                
        except Exception as e:
            print(f"⚠️ تحذير الموجه: {e} - سيتم إكمال البحث بالطريقة العادية.")

        # 3. المسح الشامل والمتوازي (Map)
        if not self.books_data:
            yield await self.emit("token", "الكتب غير متوفرة حالياً في النظام.")
            return

        yield await self.emit("evidence", "المسح الشامل في الموسوعة المرفقة...", "Map Phase")
        
        async def search_book(book_name, data):
            try:
                res = await model.generate_content_async(
                    [data["uploaded_file"], MAP_PROMPT.format(book_name=book_name, query=query)]
                )
                text = res.text.strip()
                if text.upper() == "EMPTY" or "EMPTY" in text:
                    return None
                return f"\n=== نتائج من {book_name} ===\n{text}"
            except Exception as e:
                return None

        tasks = [search_book(b_name, b_data) for b_name, b_data in self.books_data.items()]
        map_results = await asyncio.gather(*tasks)
        
        valid_results = [res for res in map_results if res]
        
        if not valid_results:
            yield await self.emit("token", "بحثت في جميع المجلدات ولم أجد إجابة دقيقة متعلقة بسؤالك أو سياق المحادثة.")
            return

        # 4. الصياغة (Reduce)
        yield await self.emit("evidence", "تم إيجاد النصوص، جاري الصياغة النهائية...", "Reduce Phase")
        gathered_info = "\n".join(valid_results)
        
        full_answer = ""
        try:
            stream = await model.generate_content_async(
                REDUCE_PROMPT.format(query=query, gathered_info=gathered_info), stream=True
            )
            async for chunk in stream:
                if chunk.text:
                    full_answer += chunk.text
                    yield await self.emit("token", chunk.text)
        except Exception as e:
            yield await self.emit("token", f"\nخطأ أثناء الصياغة: {e}")
            return

        # 5. بناء المراجع بنظام متسامح مع أخطاء النموذج
        valid_citations = list(set(re.findall(r'\[\[(.*?)\]\]', full_answer)))
        references_map = {}
        
        for citation_content in valid_citations:
            parts = re.split(r'[،,]', citation_content, maxsplit=1)
            if len(parts) == 2:
                b_name = parts[0].strip()
                pages_part = parts[1].replace('ص', '').strip()
                
                first_num_match = re.search(r'\d+', pages_part)
                if first_num_match:
                    p_num = first_num_match.group(0)
                    
                    dict_key = citation_content 
                    
                    if b_name in self.books_data:
                        pages_list = self.books_data[b_name]["pages"]
                        found_idx = next((i for i, p in enumerate(pages_list) if p["label"] == p_num), -1)
                        
                        if found_idx != -1:
                            start_idx = max(0, found_idx - CONTEXT_NEIGHBORS)
                            end_idx = min(len(pages_list), found_idx + CONTEXT_NEIGHBORS + 1)
                            
                            bundle = []
                            for ctx_i in range(start_idx, end_idx):
                                bundle.append({
                                    "label": pages_list[ctx_i].get('real_label', pages_list[ctx_i]['label']),
                                    "text": pages_list[ctx_i]['text'],
                                    "is_target": (ctx_i == found_idx)
                                })
                            references_map[dict_key] = bundle

        refs_json = json.dumps(references_map, ensure_ascii=False)
        yield await self.emit("token", f"\n\n---REFERENCES_START---{refs_json}---REFERENCES_END---")
        print("✅ [تم] اكتمل الطلب بنجاح.")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
system_instance = WebSearchSystem()

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

@app.post("/chat_stream")
async def chat_endpoint(request: Request):
    data = await request.json()
    query = data.get("message", "")
    history = data.get("history", []) 
    return StreamingResponse(system_instance.process_stream(query, history), media_type="application/x-ndjson")

@app.post("/upload_file")
async def upload_file_endpoint(file: UploadFile = File(...)):
    try:
        content = await file.read()
        return {"filename": file.filename, "display_name": file.filename, "status": "success"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
