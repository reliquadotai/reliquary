import asyncio
import httpx
import pytest
from starlette.responses import JSONResponse
from reliquary.validator.http_metrics import HttpMetrics, HttpMetricsMiddleware


@pytest.mark.asyncio
async def test_diagnostic_overload_does_not_block_submission_or_create_unbounded_labels():
    entered, release = asyncio.Event(), asyncio.Event()
    async def app(scope, receive, send):
        if scope['path'].startswith('/miner-verdicts/'):
            entered.set()
            await release.wait()
        await JSONResponse({'ok': True})(scope, receive, send)
    metrics = HttpMetrics()
    app = HttpMetricsMiddleware(app, metrics, diagnostic_limit=1)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
        first = asyncio.create_task(client.get('/miner-verdicts/one'))
        await asyncio.wait_for(entered.wait(), 1)
        rejected = await client.get('/miner-verdicts/two')
        assert rejected.status_code == 503 and rejected.headers['Retry-After'] == '1'
        assert (await client.post('/submit')).status_code == 200
        release.set()
        assert (await first).status_code == 200
        for i in range(10):
            await client.get(f'/unknown-{i}')
    assert metrics.diagnostic_active == 0
    assert set(metrics.routes) == {'GET verdict_feed', 'POST /submit', 'GET other'}
    assert metrics.routes['GET verdict_feed']['overload_rejections'] == 1
