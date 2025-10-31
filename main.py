from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict, HttpUrl
from typing import List, Optional
from pathlib import Path
from datetime import datetime, timezone
import aiohttp
import aiofiles
import logging
import uuid
import os
import re
import m3u8

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

app = FastAPI(title="IPTV M3U Playlist API", version="2.0")
api_router = APIRouter(prefix="/api")

mongo_url = os.environ.get("MONGO_URL", "mongodb://mongo:27017")
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get("DB_NAME", "iptv_db")]

class Channel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    url: str
    logo: Optional[str] = None
    group: Optional[str] = "Uncategorized"
    tvg_id: Optional[str] = None
    tvg_name: Optional[str] = None

class Playlist(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    source_type: str
    source: str
    channels: List[Channel] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PlaylistCreate(BaseModel):
    name: str
    source_type: str
    source: HttpUrl | str

class Favorite(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    playlist_id: str
    channel: Channel
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class FavoriteCreate(BaseModel):
    playlist_id: str
    channel: Channel

async def parse_m3u_content(content: str) -> List[Channel]:
    channels = []
    lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#EXTINF:"):
            info = {}
            info["tvg_id"] = _extract(line, r'tvg-id="([^"]+)"')
            info["tvg_name"] = _extract(line, r'tvg-name="([^"]+)"')
            info["logo"] = _extract(line, r'tvg-logo="([^"]+)"')
            info["group"] = _extract(line, r'group-title="([^"]+)"') or "Uncategorized"
            name_match = re.search(r",(.+)$", line)
            info["name"] = name_match.group(1).strip() if name_match else "Unknown Channel"
            if i + 1 < len(lines):
                url = lines[i + 1]
                if not url.startswith("#"):
                    channel = Channel(
                        name=info["name"],
                        url=url,
                        logo=info["logo"],
                        group=info["group"],
                        tvg_id=info["tvg_id"],
                        tvg_name=info["tvg_name"],
                    )
                    channels.append(channel)
                    i += 2
                    continue
        i += 1
    return channels

def _extract(line: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, line)
    return match.group(1) if match else None

async def validate_m3u_url(url: str) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    return False
                text = await resp.text()
                if not text.startswith("#EXTM3U"):
                    return False
                return True
    except Exception:
        return False

@api_router.get("/")
async def root():
    return {"message": "IPTV M3U Player API v2.0"}

@api_router.post("/playlists", response_model=Playlist)
async def create_playlist(playlist_data: PlaylistCreate):
    if playlist_data.source_type == "url":
        if not str(playlist_data.source).lower().startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail="Invalid URL source")
        if not await validate_m3u_url(str(playlist_data.source)):
            raise HTTPException(status_code=400, detail="Invalid or inaccessible M3U URL")
        async with aiohttp.ClientSession() as session:
            async with session.get(str(playlist_data.source)) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail="Failed to fetch M3U content")
                content = await response.text()
    else:
        raise HTTPException(status_code=400, detail="Use /playlists/upload for file uploads")

    channels = await parse_m3u_content(content)
    if not channels:
        raise HTTPException(status_code=400, detail="No valid channels found in M3U")

    playlist = Playlist(
        name=playlist_data.name,
        source_type="url",
        source=str(playlist_data.source),
        channels=channels,
    )

    await db.playlists.insert_one(playlist.model_dump())
    return playlist

@api_router.post("/playlists/upload", response_model=Playlist)
async def upload_playlist(name: str, file: UploadFile = File(...)):
    if not file.filename.endswith((".m3u", ".m3u8")):
        raise HTTPException(status_code=400, detail="Only .m3u or .m3u8 files supported")
    tmp_path = Path("/tmp") / file.filename
    async with aiofiles.open(tmp_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)
    text = content.decode("utf-8", errors="ignore")
    if not text.startswith("#EXTM3U"):
        raise HTTPException(status_code=400, detail="Invalid M3U file format")
    channels = await parse_m3u_content(text)
    if not channels:
        raise HTTPException(status_code=400, detail="No valid channels found")
    playlist = Playlist(name=name, source_type="file", source=file.filename, channels=channels)
    await db.playlists.insert_one(playlist.model_dump())
    return playlist

@api_router.get("/playlists", response_model=List[Playlist])
async def get_playlists():
    playlists = await db.playlists.find({}, {"_id": 0}).to_list(1000)
    return playlists

@api_router.get("/playlists/{playlist_id}", response_model=Playlist)
async def get_playlist(playlist_id: str):
    playlist = await db.playlists.find_one({"id": playlist_id}, {"_id": 0})
    if not playlist:
        raise HTTPException(status_code=404, detail="Playlist not found")
    return playlist

@api_router.delete("/playlists/{playlist_id}")
async def delete_playlist(playlist_id: str):
    result = await db.playlists.delete_one({"id": playlist_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Playlist not found")
    await db.favorites.delete_many({"playlist_id": playlist_id})
    return {"message": "Playlist deleted successfully"}

@api_router.get("/channels")
async def get_channels(
    search: Optional[str] = None,
    group: Optional[str] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
):
    playlists = await db.playlists.find({}, {"_id": 0}).to_list(1000)
    channels = []
    for pl in playlists:
        for ch in pl.get("channels", []):
            ch["playlist_id"] = pl["id"]
            channels.append(ch)
    if search:
        search_lower = search.lower()
        channels = [ch for ch in channels if search_lower in ch["name"].lower()]
    if group:
        channels = [ch for ch in channels if ch.get("group", "").lower() == group.lower()]
    start = (page - 1) * limit
    end = start + limit
    return {"total": len(channels), "page": page, "results": channels[start:end]}

@api_router.get("/groups")
async def get_groups():
    playlists = await db.playlists.find({}, {"_id": 0}).to_list(1000)
    groups = {ch.get("group") for pl in playlists for ch in pl.get("channels", []) if ch.get("group")}
    return sorted(groups)

@api_router.post("/favorites", response_model=Favorite)
async def add_favorite(favorite_data: FavoriteCreate):
    existing = await db.favorites.find_one(
        {"playlist_id": favorite_data.playlist_id, "channel.id": favorite_data.channel.id}
    )
    if existing:
        raise HTTPException(status_code=400, detail="Already in favorites")
    favorite = Favorite(playlist_id=favorite_data.playlist_id, channel=favorite_data.channel)
    await db.favorites.insert_one(favorite.model_dump())
    return favorite

@api_router.get("/favorites")
async def get_favorites(page: int = Query(1, ge=1), limit: int = Query(50, ge=1, le=200)):
    favorites = await db.favorites.find({}, {"_id": 0}).to_list(1000)
    start = (page - 1) * limit
    end = start + limit
    return {"total": len(favorites), "page": page, "results": favorites[start:end]}

@api_router.delete("/favorites/{favorite_id}")
async def delete_favorite(favorite_id: str):
    result = await db.favorites.delete_one({"id": favorite_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Favorite not found")
    return {"message": "Favorite deleted"}

@api_router.delete("/favorites/channel/{channel_id}")
async def delete_favorite_by_channel(channel_id: str):
    result = await db.favorites.delete_one({"channel.id": channel_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Favorite not found")
    return {"message": "Favorite deleted"}

@api_router.get("/health")
async def health():
    try:
        await db.command("ping")
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database not reachable: {e}")

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("iptv_api")

@app.on_event("startup")
async def startup_db():
    await db.playlists.create_index("id", unique=True)
    await db.favorites.create_index("id", unique=True)
    await db.favorites.create_index("channel.id")

@app.on_event("shutdown")
async def shutdown_db():
    client.close()
