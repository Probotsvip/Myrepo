import os
import re
import json
import base64
import asyncio
import logging
import secrets
import random
import datetime
from typing import Optional, Dict, Any, List, Tuple
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI, Request, HTTPException, Query, Form
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pyrogram import Client
from pyrogram.enums import ParseMode
from motor.motor_asyncio import AsyncIOMotorClient
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

# -------------------
# Enhanced logger setup
# -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger("yt_api_pro")

# -------------------
# Config (from env)
# -------------------
API_ID = int(os.getenv("API_ID", "22947426"))
API_HASH = os.getenv("API_HASH", "3ac67de232f419724b3c905e1934bc7b")
BOT_TOKEN = os.getenv("BOT_TOKEN", "8495258927:AAEG9JBCRagfOHnpbYBFyMeMmjDLnCsyAoQ")
CACHE_CHANNEL_ID = int(os.getenv("CACHE_CHANNEL_ID", "-1003086955999"))
MONGO_DB_URI = os.getenv("MONGO_DB_URI", "mongodb+srv://jaydipmore74:xCpTm5OPAfRKYnif@cluster0.5jo18.mongodb.net/?retryWrites=true&w=majority")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "111222")
DEFAULT_VIDEO_QUALITY = os.getenv("DEFAULT_VIDEO_QUALITY", "720")
DEFAULT_AUDIO_QUALITY = os.getenv("DEFAULT_AUDIO_QUALITY", "320")
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "./downloads")
MAX_CONCURRENT_DOWNLOADS = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "3"))
SAVETUBE_BASE = "https://media.savetube.me"

# Validate required environment variables
if not all([API_ID, API_HASH, BOT_TOKEN, CACHE_CHANNEL_ID, MONGO_DB_URI]):
    logger.error("Missing required environment variables")
    raise RuntimeError("Please set all required environment variables")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# -------------------
# Database connections
# -------------------
try:
    mongo_client = AsyncIOMotorClient(MONGO_DB_URI)
    mongodb = mongo_client.yt_api
    logger.info("Connected to MongoDB successfully")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise

# -------------------
# Pyrogram client with enhanced reliability
# -------------------
pyrogram_client = Client(
    "yt_cache_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN,
    in_memory=True
)

# Telegram connection health monitor
TELEGRAM_CONNECTED = False

# -------------------
# Constants and utilities
# -------------------
YT_RE = re.compile(
    r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|embed/|shorts/)|youtu\.be/)?([0-9A-Za-z_-]{11})'
)

# AES key for Savetube decryption (from original JS)
AES_HEX_KEY = "C5D58EF67A7584E4A29F6C35BBC4EB12"
AES_KEY_BYTES = bytes.fromhex(AES_HEX_KEY)

# Connection pool for aiohttp
session_pool = None

# -------------------
# Enhanced helper functions
# -------------------
@asynccontextmanager
async def get_session():
    """Get an aiohttp session from pool"""
    global session_pool
    if session_pool is None:
        session_pool = aiohttp.ClientSession()
    
    yield session_pool

async def close_session_pool():
    """Close the session pool"""
    global session_pool
    if session_pool:
        await session_pool.close()
        session_pool = None

def ist_date_str() -> str:
    """Get current date string in IST"""
    now_ist = datetime.datetime.now(ZoneInfo("Asia/Kolkata"))
    return now_ist.strftime("%Y-%m-%d")

def decrypt_savetube_data(b64_encrypted: str) -> dict:
    """Decrypt Savetube response data"""
    try:
        raw = base64.b64decode(b64_encrypted)
        iv = raw[:16]
        ciphertext = raw[16:]
        cipher = AES.new(AES_KEY_BYTES, AES.MODE_CBC, iv=iv)
        decrypted = cipher.decrypt(ciphertext)
        unpadded = unpad(decrypted, 16, style='pkcs7')
        return json.loads(unpadded.decode())
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise

def normalize_youtube_input(inp: str) -> str:
    """Normalize YouTube URL or ID to full URL"""
    inp = inp.strip()
    m = YT_RE.search(inp)
    if m:
        vidid = m.group(1)
        return f"https://www.youtube.com/watch?v={vidid}"
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", inp):
        return f"https://www.youtube.com/watch?v={inp}"
    return inp

def extract_vidid(inp: str) -> Optional[str]:
    """Extract YouTube video ID from input"""
    m = YT_RE.search(inp.strip())
    if m:
        return m.group(1)
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", inp.strip()):
        return inp.strip()
    return None

def sanitize_filename(name: str) -> str:
    """Sanitize filename to remove invalid characters"""
    return re.sub(r'[^\w\-_\. ]', '_', name)

async def get_random_cdn(session: aiohttp.ClientSession) -> str:
    """Get a random CDN from Savetube"""
    try:
        async with session.get(f"{SAVETUBE_BASE}/api/random-cdn", timeout=10) as r:
            js = await r.json()
            return js.get("cdn", "cdn1.savetube.me")
    except Exception:
        return "cdn1.savetube.me"  # Fallback

async def savetube_info(session: aiohttp.ClientSession, youtube_url: str) -> dict:
    """Get video info from Savetube"""
    cdn = await get_random_cdn(session)
    info_url = f"https://{cdn}/v2/info"
    
    try:
        async with session.post(info_url, json={"url": youtube_url}, timeout=15) as r:
            js = await r.json()
            if not js.get("status"):
                raise Exception(js.get("message", "Failed to fetch video info"))
            
            result = decrypt_savetube_data(js["data"])
            return result
    except Exception as e:
        logger.error(f"Savetube info error: {e}")
        raise

async def savetube_download(session: aiohttp.ClientSession, key: str, quality: str, 
                           download_type: str = "video") -> str:
    """Get download URL from Savetube"""
    cdn = await get_random_cdn(session)
    download_url = f"https://{cdn}/download"
    
    payload = {
        "downloadType": download_type,
        "quality": quality,
        "key": key
    }
    
    try:
        async with session.post(download_url, json=payload, timeout=20) as r:
            js = await r.json()
            if js.get("status") and js.get("data", {}).get("downloadUrl"):
                return js["data"]["downloadUrl"]
            raise Exception(js.get("message", "Failed to get download URL"))
    except Exception as e:
        logger.error(f"Savetube download error: {e}")
        raise

def choose_best_quality(decrypted: dict, media_type: str) -> str:
    """Choose the best available quality"""
    if media_type == "video":
        formats = decrypted.get("formats", [])
        video_formats = [f for f in formats if f.get("type") == "video"]
        if video_formats:
            video_formats.sort(key=lambda x: x.get("height", 0), reverse=True)
            return str(video_formats[0].get("height", DEFAULT_VIDEO_QUALITY))
        return DEFAULT_VIDEO_QUALITY
    else:
        formats = decrypted.get("formats", [])
        audio_formats = [f for f in formats if f.get("type") == "audio"]
        if audio_formats:
            audio_formats.sort(key=lambda x: x.get("bitrate", 0), reverse=True)
            return str(audio_formats[0].get("bitrate", DEFAULT_AUDIO_QUALITY))
        return DEFAULT_AUDIO_QUALITY

async def get_cached_file(ytid: str, media_type: str) -> Optional[Dict]:
    """Check if file is cached in MongoDB"""
    cache_key = f"{ytid}:{media_type}"
    
    # Check MongoDB
    doc = await mongodb.cache.find_one({"ytid": ytid, "type": media_type})
    if doc:
        return doc
    
    return None

async def save_cache_record(ytid: str, media_type: str, file_id: str, 
                           chat_id: int, msg_id: int, file_name: str, meta: dict):
    """Save cache record to MongoDB"""
    doc = {
        "ytid": ytid,
        "type": media_type,
        "file_id": file_id,
        "chat_id": chat_id,
        "msg_id": msg_id,
        "file_name": file_name,
        "meta": meta,
        "cached_at": datetime.datetime.utcnow()
    }
    
    await mongodb.cache.update_one(
        {"ytid": ytid, "type": media_type},
        {"$set": doc},
        upsert=True
    )
    
    return doc

async def check_api_key(key: str):
    """Validate API key and check usage limits"""
    if not key:
        raise HTTPException(status_code=401, detail="API key required")
    
    record = await mongodb.apikeys.find_one({"key": key})
    if not record:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Check expiry
    expiry = record.get("expiry_date")
    if expiry:
        if isinstance(expiry, str):
            expiry = datetime.datetime.fromisoformat(expiry)
        if expiry < datetime.datetime.utcnow():
            raise HTTPException(status_code=403, detail="API key expired")
    
    # Check daily usage (IST date)
    today = ist_date_str()
    if record.get("last_used_date") != today:
        # Reset daily counter
        await mongodb.apikeys.update_one(
            {"key": key},
            {"$set": {"used_today": 0, "last_used_date": today}}
        )
        record["used_today"] = 0
    
    used = record.get("used_today", 0)
    limit = record.get("daily_limit", 1000)
    
    if used >= limit:
        raise HTTPException(status_code=429, detail="Daily limit reached")
    
    # Increment usage counter
    await mongodb.apikeys.update_one(
        {"key": key},
        {"$inc": {"used_today": 1}}
    )
    
    return record

async def create_api_key(owner: str = "user", daily_limit: int = 1000, 
                        days_valid: int = 30, is_admin: bool = False):
    """Create a new API key"""
    key = secrets.token_urlsafe(32)
    expiry = None
    
    if days_valid:
        expiry = (datetime.datetime.utcnow() + datetime.timedelta(days=days_valid))
    
    doc = {
        "key": key,
        "owner": owner,
        "daily_limit": daily_limit,
        "used_today": 0,
        "last_used_date": None,
        "expiry_date": expiry,
        "is_admin": is_admin,
        "created_at": datetime.datetime.utcnow()
    }
    
    await mongodb.apikeys.insert_one(doc)
    return doc

async def download_file(session: aiohttp.ClientSession, url: str, dest_path: str):
    """Download a file with progress tracking and retry mechanism"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with session.get(url, timeout=300) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                with open(dest_path, 'wb') as f:
                    downloaded = 0
                    async for chunk in response.content.iter_chunked(1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0 and int((downloaded / total_size) * 100) % 10 == 0:
                            logger.info(f"Downloaded {downloaded}/{total_size} bytes of {dest_path}")
                
                logger.info(f"Successfully downloaded {dest_path}")
                return
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

async def upload_to_telegram(file_path: str, title: str, media_type: str):
    """Upload file to Telegram channel with retry mechanism"""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            if not TELEGRAM_CONNECTED:
                await ensure_telegram_connection()
                
            if media_type == "video":
                sent_message = await pyrogram_client.send_video(
                    chat_id=CACHE_CHANNEL_ID,
                    video=file_path,
                    caption=title[:1000],
                    parse_mode=ParseMode.HTML
                )
                file_id = sent_message.video.file_id
            else:
                sent_message = await pyrogram_client.send_audio(
                    chat_id=CACHE_CHANNEL_ID,
                    audio=file_path,
                    caption=title[:1000],
                    parse_mode=ParseMode.HTML,
                    title=title[:64],
                    performer="YouTube"
                )
                file_id = sent_message.audio.file_id
            
            return file_id, sent_message.id
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Upload attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2 ** attempt)

async def ensure_telegram_connection():
    """Ensure Telegram connection is active"""
    global TELEGRAM_CONNECTED
    if not TELEGRAM_CONNECTED:
        try:
            if not pyrogram_client.is_connected:
                await pyrogram_client.start()
            TELEGRAM_CONNECTED = True
            logger.info("Telegram connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Telegram: {e}")
            raise

async def get_telegram_download_url(file_id: str) -> str:
    """Get direct download URL for Telegram file"""
    try:
        await ensure_telegram_connection()
        file_info = await pyrogram_client.get_file(file_id)
        return f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}"
    except Exception as e:
        logger.error(f"Failed to get Telegram download URL: {e}")
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.telegram.org/bot{BOT_TOKEN}/getFile?file_id={file_id}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("ok") and data.get("result", {}).get("file_path"):
                            file_path = data["result"]["file_path"]
                            return f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
        except Exception as e2:
            logger.error(f"Fallback method also failed: {e2}")
        
        return f"https://api.telegram.org/bot{BOT_TOKEN}/getFile?file_id={file_id}"

# -------------------
# God-Level Uploader System
# -------------------
class GodUploader:
    """Ultra-reliable uploader system with guaranteed Telegram delivery"""
    
    def __init__(self):
        self.task_queue = asyncio.Queue()
        self.processing_tasks = set()
        self.failed_tasks = set()
        self.retry_delay = 300  # 5 minutes between retries
        self.max_retries = 10
        self.active_workers = 0
        self.max_workers = MAX_CONCURRENT_DOWNLOADS
        self.upload_stats = {
            'successful': 0,
            'failed': 0,
            'retries': 0,
            'queue_size': 0
        }
        
    async def start_workers(self):
        """Start worker tasks for processing uploads"""
        for _ in range(self.max_workers):
            asyncio.create_task(self.upload_worker())
            
    async def add_task(self, task_data: dict):
        """Add a new upload task to queue"""
        cache_key = f"{task_data['ytid']}:{task_data['media_type']}"
        
        # Check if already in queue or processing
        if cache_key in self.processing_tasks or cache_key in self.failed_tasks:
            return False
            
        await self.task_queue.put(task_data)
        self.processing_tasks.add(cache_key)
        self.upload_stats['queue_size'] = self.task_queue.qsize()
        return True
        
    async def upload_worker(self):
        """Worker task that processes uploads from queue"""
        self.active_workers += 1
        logger.info(f"Upload worker started. Total workers: {self.active_workers}")
        
        while True:
            try:
                task_data = await self.task_queue.get()
                cache_key = f"{task_data['ytid']}:{task_data['media_type']}"
                
                # Process the task with retry mechanism
                success = await self.process_with_retries(task_data)
                
                if success:
                    self.upload_stats['successful'] += 1
                    logger.info(f"Successfully processed {cache_key}")
                else:
                    self.upload_stats['failed'] += 1
                    logger.error(f"Permanently failed to process {cache_key}")
                    
                self.processing_tasks.discard(cache_key)
                self.upload_stats['queue_size'] = self.task_queue.qsize()
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Upload worker crashed: {e}")
                await asyncio.sleep(5)  # Prevent tight loop on errors
                
    async def process_with_retries(self, task_data: dict) -> bool:
        """Process task with exponential backoff retry"""
        cache_key = f"{task_data['ytid']}:{task_data['media_type']}"
        
        for attempt in range(self.max_retries):
            try:
                # Check cache again before processing
                cached = await get_cached_file(task_data['ytid'], task_data['media_type'])
                if cached:
                    logger.info(f"Already cached during retry: {cache_key}")
                    return True
                    
                # Download and upload
                async with get_session() as session:
                    # Download file
                    ext = "mp4" if task_data['media_type'] == "video" else "mp3"
                    safe_title = sanitize_filename(task_data['title'])[:100]
                    filename = f"{task_data['ytid']}_{task_data['media_type']}_{task_data['quality']}.{ext}"
                    filepath = os.path.join(DOWNLOAD_DIR, filename)
                    
                    await download_file(session, task_data['download_url'], filepath)
                    
                    # Upload to Telegram
                    file_id, msg_id = await upload_to_telegram(
                        filepath, task_data['title'], task_data['media_type']
                    )
                    
                    # Save to cache
                    await save_cache_record(
                        task_data['ytid'], task_data['media_type'], file_id,
                        CACHE_CHANNEL_ID, msg_id, filename, task_data['meta']
                    )
                    
                # Clean up
                try:
                    os.remove(filepath)
                except:
                    pass
                    
                return True
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {cache_key}: {e}")
                self.upload_stats['retries'] += 1
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    delay = self.retry_delay * (2 ** attempt) * (0.5 + random.random())
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed, add to failed set
                    self.failed_tasks.add(cache_key)
                    return False
                    
    async def get_stats(self) -> dict:
        """Get current upload system statistics"""
        return {
            **self.upload_stats,
            'active_workers': self.active_workers,
            'processing_tasks': len(self.processing_tasks),
            'failed_tasks': len(self.failed_tasks)
        }

# Initialize the uploader system
uploader = GodUploader()

async def guaranteed_upload(ytid: str, media_type: str, download_url: str, 
                           title: str, quality: str, meta: dict):
    """Guaranteed upload to Telegram via uploader system"""
    task_data = {
        'ytid': ytid,
        'media_type': media_type,
        'download_url': download_url,
        'title': title,
        'quality': quality,
        'meta': meta
    }
    
    # Add to uploader queue
    added = await uploader.add_task(task_data)
    
    if not added:
        logger.info(f"Task already in queue: {ytid}:{media_type}")
    else:
        logger.info(f"Task added to queue: {ytid}:{media_type}")

# -------------------
# API endpoints with enhanced performance
# -------------------
class ResultModel(BaseModel):
    status: bool
    creator: str = "@Nottyboyy"
    telegram: str = "https://t.me/ZeeMusicUpdate"
    result: Optional[dict] = None
    message: Optional[str] = None

# Initialize FastAPI with lifespan events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await ensure_telegram_connection()
    
    # Create MongoDB indexes
    await mongodb.cache.create_index("ytid")
    await mongodb.cache.create_index([("ytid", 1), ("type", 1)])
    await mongodb.apikeys.create_index("key", unique=True)
    await mongodb.requests.create_index("timestamp")
    
    # Start the uploader workers
    await uploader.start_workers()
    logger.info("GodUploader system started")
    
    # Create a default admin key if none exists
    count = await mongodb.apikeys.count_documents({})
    if count == 0:
        key_data = await create_api_key("admin", 10000, 365, True)
        logger.info(f"Created default admin key: {key_data['key']}")
    
    yield
    
    # Shutdown
    await close_session_pool()
    if TELEGRAM_CONNECTED:
        await pyrogram_client.stop()
        logger.info("Pyrogram client stopped")

app = FastAPI(title="Ultra Pro YouTube to MP3/MP4 API", lifespan=lifespan)

# Mount static files for admin panel
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to docs"""
    return RedirectResponse(url="/docs")

@app.get("/ytmp4", response_model=ResultModel)
async def ytmp4(
    request: Request,
    url: str = Query(..., description="YouTube URL or video ID"),
    api_key: str = Query(..., description="Your API key"),
    quality: str = Query(None, description="Preferred quality (e.g., 720, 1080)")
):
    """Convert YouTube video to MP4 with enhanced performance"""
    # Validate API key
    await check_api_key(api_key)
    
    # Normalize YouTube URL
    yt_url = normalize_youtube_input(url)
    vidid = extract_vidid(url) or extract_vidid(yt_url)
    
    if not vidid:
        return JSONResponse(
            status_code=400,
            content={
                "status": False,
                "message": "Invalid YouTube URL or ID",
                "creator": "@Nottyboyy",
                "telegram": "https://t.me/ZeeMusicUpdate"
            }
        )
    
    # Check cache first
    cached = await get_cached_file(vidid, "video")
    if cached:
        telegram_url = await get_telegram_download_url(cached["file_id"])
        tlink = f"https://t.me/c/{str(cached['chat_id']).replace('-100', '')}/{cached['msg_id']}"
        
        return {
            "status": True,
            "creator": "@Nottyboyy",
            "telegram": "https://t.me/ZeeMusicUpdate",
            "result": {
                "title": cached["meta"].get("title", "Unknown"),
                "duration": cached["meta"].get("duration", "Unknown"),
                "quality": cached["meta"].get("quality", "Unknown"),
                "source": "telegram_cache",
                "url": telegram_url,
                "file_id": cached["file_id"],
                "telegram_msg": {
                    "chat_id": cached["chat_id"],
                    "msg_id": cached["msg_id"],
                    "tlink": tlink
                }
            }
        }
    
    # Not cached, use Savetube with connection pooling
    async with get_session() as session:
        try:
            # Get video info
            decrypted = await savetube_info(session, yt_url)
            
            # Determine quality
            selected_quality = quality or choose_best_quality(decrypted, "video")
            
            # Get download URL
            download_url = await savetube_download(
                session, decrypted.get("key"), selected_quality, "video"
            )
            
            # Prepare response
            response_data = {
                "status": True,
                "creator": "@Nottyboyy",
                "telegram": "https://t.me/ZeeMusicUpdate",
                "result": {
                    "title": decrypted.get("title", "Unknown"),
                    "duration": decrypted.get("durationLabel", "Unknown"),
                    "quality": selected_quality,
                    "source": "savetube",
                    "url": download_url
                }
            }
            
            # Start guaranteed upload to Telegram
            await guaranteed_upload(
                vidid, "video", download_url,
                decrypted.get("title", vidid),
                selected_quality,
                {
                    "duration": decrypted.get("durationLabel", "Unknown"),
                    "thumbnail": decrypted.get("thumbnail")
                }
            )
            
            return response_data
            
        except Exception as e:
            logger.error(f"YTMP4 error: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": False,
                    "message": str(e),
                    "creator": "@Nottyboyy",
                    "telegram": "https://t.me/ZeeMusicUpdate"
                }
            )

@app.get("/ytmp3", response_model=ResultModel)
async def ytmp3(
    request: Request,
    url: str = Query(..., description="YouTube URL or video ID"),
    api_key: str = Query(..., description="Your API key"),
    quality: str = Query(None, description="Preferred quality (e.g., 128, 320)")
):
    """Convert YouTube video to MP3 with enhanced performance"""
    # Validate API key
    await check_api_key(api_key)
    
    # Normalize YouTube URL
    yt_url = normalize_youtube_input(url)
    vidid = extract_vidid(url) or extract_vidid(yt_url)
    
    if not vidid:
        return JSONResponse(
            status_code=400,
            content={
                "status": False,
                "message": "Invalid YouTube URL or ID",
                "creator": "@Nottyboyy",
                "telegram": "https://t.me/ZeeMusicUpdate"
            }
        )
    
    # Check cache first
    cached = await get_cached_file(vidid, "audio")
    if cached:
        telegram_url = await get_telegram_download_url(cached["file_id"])
        tlink = f"https://t.me/c/{str(cached['chat_id']).replace('-100', '')}/{cached['msg_id']}"
        
        return {
            "status": True,
            "creator": "@Nottyboyy",
            "telegram": "https://t.me/ZeeMusicUpdate",
            "result": {
                "title": cached["meta"].get("title", "Unknown"),
                "duration": cached["meta"].get("duration", "Unknown"),
                "quality": cached["meta"].get("quality", "Unknown"),
                "source": "telegram_cache",
                "url": telegram_url,
                "file_id": cached["file_id"],
                "telegram_msg": {
                    "chat_id": cached["chat_id"],
                    "msg_id": cached["msg_id"],
                    "tlink": tlink
                }
            }
        }
    
    # Not cached, use Savetube with connection pooling
    async with get_session() as session:
        try:
            # Get video info
            decrypted = await savetube_info(session, yt_url)
            
            # Determine quality
            selected_quality = quality or choose_best_quality(decrypted, "audio")
            
            # Get download URL
            download_url = await savetube_download(
                session, decrypted.get("key"), selected_quality, "audio"
            )
            
            # Prepare response
            response_data = {
                "status": True,
                "creator": "@Nottyboyy",
                "telegram": "https://t.me/ZeeMusicUpdate",
                "result": {
                    "title": decrypted.get("title", "Unknown"),
                    "duration": decrypted.get("durationLabel", "Unknown"),
                    "quality": selected_quality,
                    "source": "savetube",
                    "url": download_url
                }
            }
            
            # Start guaranteed upload to Telegram
            await guaranteed_upload(
                vidid, "audio", download_url,
                decrypted.get("title", vidid),
                selected_quality,
                {
                    "duration": decrypted.get("durationLabel", "Unknown"),
                    "thumbnail": decrypted.get("thumbnail")
                }
            )
            
            return response_data
            
        except Exception as e:
            logger.error(f"YTMP3 error: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": False,
                    "message": str(e),
                    "creator": "@Nottyboyy",
                    "telegram": "https://t.me/ZeeMusicUpdate"
                }
            )

@app.post("/admin/create_key")
async def admin_create_key(
    request: Request,
    owner: str = Form("user"),
    daily_limit: int = Form(1000),
    days_valid: int = Form(30),
    is_admin: bool = Form(False)
):
    """Create a new API key (admin only)"""
    # Check admin secret
    admin_secret = request.headers.get("X-Admin-Secret")
    if admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Invalid admin secret")
    
    # Create the key
    key_data = await create_api_key(owner, daily_limit, days_valid, is_admin)
    
    return {
        "status": True,
        "creator": "@Nottyboyy",
        "telegram": "https://t.me/ZeeMusicUpdate",
        "result": {
            "key": key_data["key"],
            "owner": key_data["owner"],
            "daily_limit": key_data["daily_limit"],
            "expires": key_data.get("expiry_date"),
            "is_admin": key_data["is_admin"]
        }
    }

@app.get("/admin/stats")
async def admin_stats(request: Request):
    """Get API usage statistics (admin only)"""
    # Check admin secret
    admin_secret = request.headers.get("X-Admin-Secret")
    if admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Invalid admin secret")
    
    # Get stats
    total_keys = await mongodb.apikeys.count_documents({})
    total_cached = await mongodb.cache.count_documents({})
    today = ist_date_str()
    
    # Today's usage
    today_usage = 0
    keys = await mongodb.apikeys.find({"last_used_date": today}).to_list(None)
    for key in keys:
        today_usage += key.get("used_today", 0)
    
    # Upload system status
    upload_stats = await uploader.get_stats()
    
    return {
        "status": True,
        "creator": "@Nottyboyy",
        "telegram": "https://t.me/ZeeMusicUpdate",
        "result": {
            "total_keys": total_keys,
            "total_cached": total_cached,
            "today_usage": today_usage,
            "upload_system": upload_stats
        }
    }

# -------------------
# Admin Panel UI Routes
# -------------------
@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    """Admin panel dashboard"""
    admin_secret = request.headers.get("X-Admin-Secret") or request.query_params.get("key")
    if admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Invalid admin secret")
    
    # Get basic stats
    total_keys = await mongodb.apikeys.count_documents({})
    total_cached = await mongodb.cache.count_documents({})
    today = ist_date_str()
    
    today_usage = 0
    keys = await mongodb.apikeys.find({"last_used_date": today}).to_list(None)
    for key in keys:
        today_usage += key.get("used_today", 0)
    
    # Get uploader stats
    upload_stats = await uploader.get_stats()
    
    # Recent API keys
    recent_keys = await mongodb.apikeys.find().sort("created_at", -1).limit(5).to_list(None)
    
    # Recent cached items
    recent_cached = await mongodb.cache.find().sort("cached_at", -1).limit(5).to_list(None)
    
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "total_keys": total_keys,
        "total_cached": total_cached,
        "today_usage": today_usage,
        "recent_keys": recent_keys,
        "recent_cached": recent_cached,
        "upload_stats": upload_stats,
        "system_status": {
            "telegram_connected": TELEGRAM_CONNECTED,
            "concurrent_downloads": upload_stats['active_workers']
        }
    })

@app.get("/admin/keys", response_class=HTMLResponse)
async def admin_keys(request: Request):
    """API keys management"""
    admin_secret = request.headers.get("X-Admin-Secret") or request.query_params.get("key")
    if admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Invalid admin secret")
    
    keys = await mongodb.apikeys.find().sort("created_at", -1).to_list(None)
    return templates.TemplateResponse("keys.html", {
        "request": request,
        "keys": keys
    })

@app.get("/admin/cache", response_class=HTMLResponse)
async def admin_cache(request: Request):
    """Cache management"""
    admin_secret = request.headers.get("X-Admin-Secret") or request.query_params.get("key")
    if admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Invalid admin secret")
    
    page = int(request.query_params.get("page", 1))
    per_page = 20
    skip = (page - 1) * per_page
    
    total = await mongodb.cache.count_documents({})
    cached_items = await mongodb.cache.find().sort("cached_at", -1).skip(skip).limit(per_page).to_list(None)
    
    return templates.TemplateResponse("cache.html", {
        "request": request,
        "cached_items": cached_items,
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": (total + per_page - 1) // per_page
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 6000))
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=300)