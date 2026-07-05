import pytest


class FakeDehydrator:
    async def analyze(self, content):
        return {
            "domain": ["journal"],
            "valence": 0.5,
            "arousal": 0.3,
            "tags": ["diary"],
            "suggested_name": "daily-note",
            "importance": 5,
        }

    async def digest(self, content):
        return [
            {
                "content": content,
                "tags": ["diary"],
                "importance": 5,
                "domain": ["journal"],
                "valence": 0.5,
                "arousal": 0.3,
                "name": "daily-note",
            }
        ]

    async def merge(self, old_content, new_content):
        return old_content + "\n" + new_content

    async def dehydrate(self, content, metadata):
        return content


class FakeDecayEngine:
    is_running = False

    async def ensure_started(self):
        return None


class FakeEmbeddingEngine:
    enabled = False

    async def generate_and_store(self, bucket_id, content):
        return None

    def delete_embedding(self, bucket_id):
        return None


@pytest.fixture
def isolated_server(test_config, tmp_path, monkeypatch):
    import server
    from bucket_manager import BucketManager

    buckets_dir = str(tmp_path / "buckets")
    cfg = dict(test_config)
    cfg["buckets_dir"] = buckets_dir
    cfg["auto_merge"] = False

    monkeypatch.setitem(server.config, "buckets_dir", buckets_dir)
    monkeypatch.setitem(server.config, "auto_merge", False)
    monkeypatch.setattr(server, "bucket_mgr", BucketManager(cfg))
    monkeypatch.setattr(server, "dehydrator", FakeDehydrator())
    monkeypatch.setattr(server, "decay_engine", FakeDecayEngine())
    monkeypatch.setattr(server, "embedding_engine", FakeEmbeddingEngine())
    return server


@pytest.mark.asyncio
async def test_grow_indexes_diary_and_read_by_title_or_date(isolated_server):
    grow = getattr(isolated_server.grow, "fn", isolated_server.grow)
    diary_list = getattr(isolated_server.diary_list, "fn", isolated_server.diary_list)
    diary_read = getattr(isolated_server.diary_read, "fn", isolated_server.diary_read)

    original = "今天修好了日记读取接口。\n以后可以按标题和日期找回正文。"
    out = await grow(content=original, event_time="2026-07-05", title="接口修复日记")

    assert "diary_id:" in out
    listed = await diary_list(date="2026-07-05")
    assert "接口修复日记" in listed

    by_title = await diary_read(title="接口修复")
    assert original in by_title

    by_date = await diary_read(date="2026-07-05")
    assert original in by_date


@pytest.mark.asyncio
async def test_trace_allows_safe_user_metadata_but_blocks_destructive(isolated_server):
    trace = getattr(isolated_server.trace, "fn", isolated_server.trace)
    bucket_id = await isolated_server.bucket_mgr.create(
        content="用户手写事实正文",
        tags=[],
        importance=5,
        domain=["manual"],
        valence=0.5,
        arousal=0.3,
        name="old-name",
        created_by="user",
    )

    safe = await trace(bucket_id=bucket_id, name="new-name", event_time="2026-07-05")
    assert "已修改记忆桶" in safe
    updated = await isolated_server.bucket_mgr.get(bucket_id)
    assert updated["metadata"]["name"] == "new-name"
    assert updated["metadata"]["event_time"].startswith("2026-07-05")

    blocked = await trace(bucket_id=bucket_id, content="改正文", delete=False)
    assert "被拦截字段: content" in blocked
    unchanged = await isolated_server.bucket_mgr.get(bucket_id)
    assert unchanged["content"] == "用户手写事实正文"

    delete_blocked = await trace(bucket_id=bucket_id, delete=True)
    assert "不能由 AI 删除" in delete_blocked


@pytest.mark.asyncio
async def test_breath_literal_fallback_finds_tags(isolated_server):
    breath = getattr(isolated_server.breath, "fn", isolated_server.breath)
    await isolated_server.bucket_mgr.create(
        content="今天记录了办公室里几位同事之间的互动。",
        tags=["同事关系", "Sandy", "Ivanka"],
        importance=8,
        domain=["work"],
        valence=0.5,
        arousal=0.3,
        name="工作与生活全记录",
        event_time="2026-06-25",
    )
    async def empty_search(*args, **kwargs):
        return []
    isolated_server.bucket_mgr.search = empty_search

    out = await breath(query="同事关系", max_results=5, max_tokens=3000)

    assert "[字面命中]" in out
    assert "同事" in out


@pytest.mark.asyncio
async def test_diary_read_date_falls_back_to_legacy_buckets(isolated_server):
    diary_read = getattr(isolated_server.diary_read, "fn", isolated_server.diary_read)
    await isolated_server.bucket_mgr.create(
        content="旧版 grow 只留下了拆分后的 6 月 25 日同事关系记录。",
        tags=["同事关系"],
        importance=8,
        domain=["work"],
        valence=0.5,
        arousal=0.3,
        name="6.25 工作记录",
        event_time="2026-06-25",
    )

    out = await diary_read(date="2026.6.25")

    assert "[legacy_date:2026-06-25]" in out
    assert "同事关系记录" in out
